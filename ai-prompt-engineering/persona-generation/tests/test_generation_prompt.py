import pytest

import generation_prompt as gp
from schema import Mode, PersonaIdentity, PersonaPersonality, ToneProfile


def make_generated_fields(**overrides) -> gp.GeneratedPersonaFields:
    defaults = dict(
        identity=PersonaIdentity(
            name="Dana Whitfield",
            role_or_title="Chief Executive Officer",
            relationship_to_user="employer",
            age_range="40-50",
            background="Has led the company for six years; budget talk makes her tense.",
        ),
        personality=PersonaPersonality(
            traits=["direct", "no-nonsense", "results-oriented"],
            communication_style=["short sentences", "redirects to process"],
            goals_in_conversation=["keep the conversation on schedule"],
            potential_triggers=["being asked for a firm number on the spot"],
        ),
        tone=ToneProfile(
            speech_register="measured, polite, face-saving",
            deflection_style="promises to follow up rather than answering directly",
            example_phrase="Great question, I'll get back to you.",
        ),
    )
    defaults.update(overrides)
    return gp.GeneratedPersonaFields(**defaults)


def test_build_generation_messages_embeds_the_right_tone_rule():
    messages = gp.build_generation_messages(
        scenario_text="Salary negotiation", persona_text="My manager Dana", mode=Mode.PERSONAL
    )
    system = messages[0]["content"]
    assert "Register for this session: personal" in system
    assert "Yaar, I don't know." in system
    assert "difficulty" in system.lower()  # explicitly told it has none


def test_build_generation_messages_rejects_empty_input():
    with pytest.raises(ValueError):
        gp.build_generation_messages("", "persona text", Mode.PROFESSIONAL)
    with pytest.raises(ValueError):
        gp.build_generation_messages("scenario text", "  ", Mode.PROFESSIONAL)


def test_output_schema_in_prompt_matches_generated_fields_shape():
    messages = gp.build_generation_messages("s", "p", Mode.PROFESSIONAL)
    system = messages[0]["content"]
    for key in ("identity", "personality", "tone", "baseline_dynamics"):
        assert key in system


def test_parse_and_assemble_round_trip():
    generated = make_generated_fields()
    raw = generated.model_dump_json()

    parsed = gp.parse_generated_fields(raw)
    assert parsed == generated

    persona = gp.assemble_persona(
        parsed, owner_user_id="user-1", display_name="My CEO", mode=Mode.PROFESSIONAL
    )
    assert persona.identity.name == "Dana Whitfield"
    assert persona.mode == Mode.PROFESSIONAL
    assert persona.baseline_dynamics.patience == 0.5  # default, since not overridden
    # assemble_persona takes no difficulty argument any more - a TypeError
    # here would mean this test itself needs updating, not the function.
