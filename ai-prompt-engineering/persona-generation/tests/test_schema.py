import pytest
from pydantic import ValidationError

from schema import (
    BaselineDynamics,
    DifficultyLevel,
    Mode,
    Persona,
    PersonaIdentity,
    PersonaPersonality,
    Scenario,
    ToneProfile,
)


def make_persona(**overrides) -> Persona:
    defaults = dict(
        owner_user_id="user-1",
        display_name="My CEO",
        identity=PersonaIdentity(
            name="Dana Whitfield",
            role_or_title="Chief Executive Officer",
            relationship_to_user="employer",
            age_range="40-50",
            background="Has led the company for six years.",
        ),
        personality=PersonaPersonality(
            traits=["direct", "skeptical"],
            communication_style=["short sentences"],
        ),
        tone=ToneProfile(
            speech_register="measured, polite",
            deflection_style="promises to follow up",
            example_phrase="I'll get back to you.",
        ),
        mode=Mode.PROFESSIONAL,
    )
    defaults.update(overrides)
    return Persona(**defaults)


def test_persona_gets_defaults_for_baseline_dynamics_and_lists():
    persona = make_persona()
    assert persona.baseline_dynamics == BaselineDynamics()
    assert persona.known_facts == []
    assert persona.behavioral_rules == []
    assert persona.version == 1
    assert persona.is_active is True


def test_baseline_dynamics_rejects_out_of_range_values():
    with pytest.raises(ValidationError):
        BaselineDynamics(patience=1.5)
    with pytest.raises(ValidationError):
        BaselineDynamics(trust=-0.1)


def test_personality_requires_at_least_one_trait():
    with pytest.raises(ValidationError):
        PersonaPersonality(traits=[], communication_style=["short sentences"])


def test_personality_caps_trait_list_length():
    with pytest.raises(ValidationError):
        PersonaPersonality(
            traits=["a", "b", "c", "d", "e", "f", "g"],  # 7 > max_length=6
            communication_style=["short sentences"],
        )


def test_editable_fields_is_excluded_from_serialization():
    persona = make_persona()
    dumped = persona.model_dump()
    assert "EDITABLE_FIELDS" not in dumped
    assert "identity" in Persona.model_fields["EDITABLE_FIELDS"].default


def test_scenario_requires_positive_duration():
    persona = make_persona()
    with pytest.raises(ValidationError):
        Scenario(
            persona_id=persona.persona_id,
            user_id="user-1",
            situation_text_raw="raw text",
            situation_summary="summary",
            user_goal="goal",
            interaction_type=Mode.PROFESSIONAL,
            difficulty=DifficultyLevel.MEDIUM,
            duration_seconds=0,
        )


def test_scenario_links_to_persona_by_id():
    persona = make_persona()
    scenario = Scenario(
        persona_id=persona.persona_id,
        user_id="user-1",
        situation_text_raw="I need a month of leave",
        situation_summary="Leave request",
        user_goal="Get approval for one month of leave",
        interaction_type=Mode.PROFESSIONAL,
        difficulty=DifficultyLevel.MEDIUM,
        duration_seconds=600,
    )
    assert scenario.persona_id == persona.persona_id
    # Scenario carries no personality/identity data - that's Persona's job.
    assert not hasattr(scenario, "personality")
