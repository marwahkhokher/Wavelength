from conversation_state import DynamicState, TurnPhase
from runtime_prompt import build_runtime_prompt
from schema import (
    DifficultyLevel,
    Mode,
    Persona,
    PersonaIdentity,
    PersonaPersonality,
    Scenario,
    ToneProfile,
)


def make_persona() -> Persona:
    return Persona(
        owner_user_id="user-1",
        display_name="My CEO",
        identity=PersonaIdentity(
            name="Dana Whitfield",
            role_or_title="CEO",
            relationship_to_user="employer",
            age_range="40-50",
            background="Six years running the company.",
        ),
        personality=PersonaPersonality(
            traits=["direct", "skeptical"], communication_style=["short sentences"]
        ),
        tone=ToneProfile(
            speech_register="measured, polite",
            deflection_style="promises to follow up",
            example_phrase="I'll get back to you.",
        ),
        behavioral_rules=["Never apologizes first.", "Always asks for a number."],
        mode=Mode.PROFESSIONAL,
    )


def make_scenario(persona: Persona) -> Scenario:
    return Scenario(
        persona_id=persona.persona_id,
        user_id="user-1",
        situation_text_raw="raw",
        situation_summary="Salary negotiation",
        user_goal="Convince Dana to approve a raise",
        interaction_type=Mode.PROFESSIONAL,
        difficulty=DifficultyLevel.MEDIUM,
        duration_seconds=600,
    )


def test_runtime_prompt_is_pure_string_assembly_no_network_or_async():
    # If this were an LLM call it would need to be awaited - it isn't.
    persona = make_persona()
    scenario = make_scenario(persona)
    state = DynamicState(receptiveness=0.45, patience=0.3, trust=0.5, defensiveness=0.1)
    prompt = build_runtime_prompt(persona, scenario, state)
    assert isinstance(prompt, str)


def test_runtime_prompt_includes_identity_traits_and_behavioral_rules():
    persona = make_persona()
    scenario = make_scenario(persona)
    state = DynamicState()
    prompt = build_runtime_prompt(persona, scenario, state)

    assert "Dana Whitfield" in prompt
    assert "direct" in prompt and "skeptical" in prompt
    assert "Never apologizes first." in prompt
    assert "Always asks for a number." in prompt
    assert scenario.situation_summary in prompt
    assert scenario.user_goal in prompt


def test_runtime_prompt_includes_current_dial_values():
    persona = make_persona()
    scenario = make_scenario(persona)
    state = DynamicState(receptiveness=0.42, patience=0.31, trust=0.58, defensiveness=0.15)
    state.turn_phase = TurnPhase.OBJECTION
    prompt = build_runtime_prompt(persona, scenario, state)

    assert "0.42" in prompt
    assert "0.31" in prompt
    assert "0.58" in prompt
    assert "0.15" in prompt
    assert "objection" in prompt


def test_runtime_prompt_handles_no_behavioral_rules_or_known_facts_gracefully():
    persona = make_persona()
    persona.behavioral_rules = []
    persona.known_facts = []
    scenario = make_scenario(persona)
    prompt = build_runtime_prompt(persona, scenario, DynamicState())
    assert "(none specified)" in prompt
    assert "(none recorded)" in prompt


def test_runtime_prompt_instructs_showing_not_naming_traits():
    persona = make_persona()
    scenario = make_scenario(persona)
    prompt = build_runtime_prompt(persona, scenario, DynamicState())
    assert "never by naming it" in prompt or "do not say" in prompt.lower()


def test_runtime_prompt_forbids_warming_below_threshold():
    persona = make_persona()
    scenario = make_scenario(persona)
    prompt = build_runtime_prompt(persona, scenario, DynamicState(receptiveness=0.3))
    assert "has NOT crossed" in prompt
    assert "Stay guarded and unconvinced" in prompt


def test_runtime_prompt_allows_warming_above_threshold():
    persona = make_persona()
    scenario = make_scenario(persona)
    prompt = build_runtime_prompt(persona, scenario, DynamicState(receptiveness=0.75))
    assert "has crossed" in prompt
    assert "may let your tone visibly warm" in prompt
