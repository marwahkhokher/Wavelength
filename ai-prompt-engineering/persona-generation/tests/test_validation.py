from generation_prompt import GeneratedPersonaFields
from schema import BaselineDynamics, PersonaIdentity, PersonaPersonality, ToneProfile
from validation import (
    build_judge_messages,
    check_contradictions,
    parse_judge_result,
    validate_deterministic,
)


def make_generated(traits, communication_style, background="Some background.") -> GeneratedPersonaFields:
    return GeneratedPersonaFields(
        identity=PersonaIdentity(
            name="Dana",
            role_or_title="CEO",
            relationship_to_user="employer",
            age_range="40-50",
            background=background,
        ),
        personality=PersonaPersonality(traits=traits, communication_style=communication_style),
        tone=ToneProfile(
            speech_register="measured", deflection_style="delays", example_phrase="I'll get back to you."
        ),
        baseline_dynamics=BaselineDynamics(),
    )


def test_clean_persona_passes():
    generated = make_generated(["direct", "skeptical"], ["short sentences"])
    result = validate_deterministic(generated)
    assert result.passed is True
    assert result.issues == []


def test_contradictory_traits_are_caught():
    generated = make_generated(["very formal", "extremely casual"], ["short sentences"])
    issues = check_contradictions(generated.personality)
    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert "formal" in issues[0].message and "casual" in issues[0].message


def test_contradiction_across_traits_and_communication_style_is_caught():
    generated = make_generated(["patient"], ["comes across as impatient in meetings"])
    issues = check_contradictions(generated.personality)
    assert any("patient" in i.message and "impatient" in i.message for i in issues)


def test_single_impatient_trait_alone_is_not_a_false_positive():
    """Regression: "patient" is a substring of "impatient" - word-boundary
    matching must not treat a lone "impatient" trait as containing "patient" too."""
    generated = make_generated(["impatient", "direct"], ["short sentences"])
    issues = check_contradictions(generated.personality)
    assert issues == []


def test_empty_background_fails_validation():
    generated = make_generated(["direct"], ["short sentences"], background="   ")
    result = validate_deterministic(generated)
    assert result.passed is False
    assert any("background" in i.message.lower() for i in result.issues)


def test_judge_messages_include_source_text_and_generated_json():
    generated = make_generated(["direct"], ["short sentences"])
    messages = build_judge_messages(generated, source_text="my manager Dana, very direct")
    system = messages[0]["content"]
    assert "my manager Dana, very direct" in system
    assert "Dana" in system  # from the generated JSON dump
    assert "prompt-injection" in system.lower() or "safety" in system.lower()


def test_parse_judge_result_normalizes_layer():
    raw = (
        '{"passed": false, "issues": [{"layer": "deterministic", '
        '"severity": "warning", "message": "seems off"}]}'
    )
    result = parse_judge_result(raw)
    assert result.passed is False
    assert result.issues[0].layer == "llm_judge"
