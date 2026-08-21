import pytest

import situation_extraction as se
from schema import Mode, Persona, PersonaIdentity, PersonaPersonality, ToneProfile


def make_persona() -> Persona:
    return Persona(
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
            traits=["direct", "skeptical"], communication_style=["short sentences"]
        ),
        tone=ToneProfile(
            speech_register="measured", deflection_style="delays", example_phrase="Great question."
        ),
        mode=Mode.PROFESSIONAL,
    )


def test_build_extraction_messages_rejects_empty_text():
    with pytest.raises(ValueError):
        se.build_extraction_messages("   ")


def test_new_persona_path_leaves_role_and_relationship_as_live_gaps():
    messages = se.build_extraction_messages("I have a meeting with my CEO about a raise.")
    system = messages[0]["content"]
    assert "No persona is selected yet" in system
    assert "other_person_role" in system


def test_known_persona_path_tells_model_not_to_re_derive_identity():
    persona = make_persona()
    messages = se.build_extraction_messages(
        "I need to request a one-month leave.", known_persona=persona
    )
    system = messages[0]["content"]
    assert "already selected" in system
    assert "My CEO" in system
    assert "Chief Executive Officer" in system
    assert "Do NOT generate a gap for" in system


def test_parse_situation_draft_round_trip():
    draft = se.SituationDraft(
        interaction_type=Mode.PROFESSIONAL,
        other_person_role="CEO",
        relationship="employee-employer",
        apparent_goal="salary increase",
        mentioned_facts=["three years tenure"],
        gaps=[
            se.InformationGap(
                field="communication_style",
                tier="useful",
                importance=4,
                question="How would you describe your CEO's communication style?",
            )
        ],
    )
    raw = draft.model_dump_json()
    parsed = se.parse_situation_draft(raw)
    assert parsed == draft
    assert parsed.gaps[0].tier == "useful"
