import pytest

from schema import DifficultyLevel, Mode, Persona, PersonaIdentity, PersonaPersonality, ToneProfile
from scenario_generation import (
    GeneratedScenarioFields,
    ScenarioMismatch,
    assemble_scenario,
    build_scenario_generation_messages,
    parse_generated_scenario,
)


def make_persona(mode=Mode.PROFESSIONAL) -> Persona:
    return Persona(
        owner_user_id="user-1",
        display_name="My CEO",
        identity=PersonaIdentity(
            name="Dana",
            role_or_title="CEO",
            relationship_to_user="employer",
            age_range="40-50",
            background="Six years at the company.",
        ),
        personality=PersonaPersonality(traits=["direct"], communication_style=["short sentences"]),
        tone=ToneProfile(
            speech_register="measured", deflection_style="delays", example_phrase="I'll follow up."
        ),
        mode=mode,
    )


def test_build_messages_rejects_empty_situation_text():
    with pytest.raises(ValueError):
        build_scenario_generation_messages("  ", persona=None)


def test_build_messages_references_existing_persona_when_reusing():
    persona = make_persona()
    messages = build_scenario_generation_messages("I need a month of leave", persona=persona)
    system = messages[0]["content"]
    assert "My CEO" in system
    assert "already established" in system


def test_build_messages_has_no_persona_context_for_new_persona():
    messages = build_scenario_generation_messages("I need a month of leave", persona=None)
    system = messages[0]["content"]
    assert "No established persona yet" in system


def test_parse_generated_scenario_round_trip():
    fields = GeneratedScenarioFields(
        situation_summary="Requesting one month of leave",
        user_goal="Get approval for the leave",
        extracted_slots={"duration": "one month"},
    )
    parsed = parse_generated_scenario(fields.model_dump_json())
    assert parsed == fields


def test_assemble_scenario_succeeds_when_modes_match():
    persona = make_persona(mode=Mode.PROFESSIONAL)
    generated = GeneratedScenarioFields(situation_summary="s", user_goal="g")
    scenario = assemble_scenario(
        generated,
        situation_text_raw="raw",
        persona=persona,
        user_id="user-1",
        interaction_type=Mode.PROFESSIONAL,
        difficulty=DifficultyLevel.MEDIUM,
        duration_seconds=600,
    )
    assert scenario.persona_id == persona.persona_id
    assert scenario.difficulty == DifficultyLevel.MEDIUM


def test_assemble_scenario_raises_on_mode_mismatch():
    persona = make_persona(mode=Mode.PROFESSIONAL)
    generated = GeneratedScenarioFields(situation_summary="s", user_goal="g")
    with pytest.raises(ScenarioMismatch):
        assemble_scenario(
            generated,
            situation_text_raw="raw",
            persona=persona,
            user_id="user-1",
            interaction_type=Mode.PERSONAL,  # mismatch: persona is PROFESSIONAL
            difficulty=DifficultyLevel.MEDIUM,
            duration_seconds=600,
        )
