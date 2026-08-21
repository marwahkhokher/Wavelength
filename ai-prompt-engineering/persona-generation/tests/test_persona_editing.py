import pytest

import persona_editing as pe
from llm_client import MockLLMClient
from schema import DifficultyLevel, Mode, Persona, PersonaIdentity, PersonaPersonality, ToneProfile


def make_persona(**overrides) -> Persona:
    defaults = dict(
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
            speech_register="measured", deflection_style="delays", example_phrase="I'll follow up."
        ),
        mode=Mode.PROFESSIONAL,
    )
    defaults.update(overrides)
    return Persona(**defaults)


# --- scope classification ---------------------------------------------------


def test_build_scope_messages_rejects_empty_text():
    with pytest.raises(ValueError):
        pe.build_scope_classification_messages("  ")


def test_parse_edit_scope_recognizes_permanent():
    assert pe.parse_edit_scope("permanent") == pe.EditScope.PERMANENT


def test_parse_edit_scope_recognizes_session():
    assert pe.parse_edit_scope("  Session\n") == pe.EditScope.SESSION


def test_parse_edit_scope_defaults_unrecognized_output_to_session():
    assert pe.parse_edit_scope("uh, both I guess?") == pe.EditScope.SESSION


# --- permanent edits ---------------------------------------------------------


def test_build_edit_messages_includes_current_persona_and_request():
    persona = make_persona()
    messages = pe.build_edit_messages(persona, "make him more friendly")
    system = messages[0]["content"]
    assert "Dana Whitfield" in system
    assert "make him more friendly" in system


def test_apply_permanent_edit_bumps_version_and_leaves_original_untouched():
    persona = make_persona()
    edited = pe.EditedPersonaFields(
        identity=persona.identity.model_copy(),
        personality=PersonaPersonality(
            traits=["warm", "approachable"], communication_style=["asks how you're doing"]
        ),
        behavioral_rules=[],
    )

    new_persona, version_record = pe.apply_permanent_edit(persona, edited, "make him friendlier")

    assert new_persona.version == persona.version + 1
    assert new_persona.personality.traits == ["warm", "approachable"]
    assert persona.version == 1  # original object never mutated
    assert persona.personality.traits == ["direct", "skeptical"]

    assert version_record.persona_id == persona.persona_id
    assert version_record.version == new_persona.version
    assert version_record.changed_by == "user_edit"
    assert version_record.change_summary == "make him friendlier"
    assert version_record.snapshot.personality.traits == ["warm", "approachable"]


def test_apply_permanent_edit_rejects_contradictory_result():
    persona = make_persona()
    edited = pe.EditedPersonaFields(
        identity=persona.identity.model_copy(),
        personality=PersonaPersonality(
            traits=["very formal", "extremely casual"], communication_style=["short sentences"]
        ),
    )
    with pytest.raises(pe.PersonaEditFailed):
        pe.apply_permanent_edit(persona, edited, "make him more relaxed")


def test_apply_permanent_edit_rejects_empty_background():
    persona = make_persona()
    edited = pe.EditedPersonaFields(
        identity=persona.identity.model_copy(update={"background": "  "}),
        personality=persona.personality,
    )
    with pytest.raises(pe.PersonaEditFailed):
        pe.apply_permanent_edit(persona, edited, "clear his background")


# --- session-level edits -----------------------------------------------------


def test_build_shift_messages_rejects_empty_text():
    with pytest.raises(ValueError):
        pe.build_shift_classification_messages("")


@pytest.mark.parametrize(
    "current,shift,expected",
    [
        (DifficultyLevel.MEDIUM, pe.DifficultyShift.HARDER, DifficultyLevel.HARD),
        (DifficultyLevel.MEDIUM, pe.DifficultyShift.EASIER, DifficultyLevel.EASY),
        (DifficultyLevel.MEDIUM, pe.DifficultyShift.UNCHANGED, DifficultyLevel.MEDIUM),
        (DifficultyLevel.HARD, pe.DifficultyShift.HARDER, DifficultyLevel.HARD),  # clamped
        (DifficultyLevel.EASY, pe.DifficultyShift.EASIER, DifficultyLevel.EASY),  # clamped
    ],
)
def test_apply_difficulty_shift(current, shift, expected):
    assert pe.apply_difficulty_shift(current, shift) == expected


def test_session_edit_never_touches_persona_object():
    persona = make_persona()
    original_traits = list(persona.personality.traits)
    pe.apply_difficulty_shift(DifficultyLevel.MEDIUM, pe.DifficultyShift.HARDER)
    assert persona.personality.traits == original_traits  # untouched, was never passed in


# --- end-to-end routing -------------------------------------------------------


async def test_route_permanent_edit_end_to_end():
    persona = make_persona()
    llm = MockLLMClient(
        responses=[
            "permanent",
            (
                '{"identity": {"name": "Dana Whitfield", "role_or_title": "CEO", '
                '"relationship_to_user": "employer", "age_range": "40-50", '
                '"background": "Six years running the company."}, '
                '"personality": {"traits": ["warm", "approachable"], '
                '"communication_style": ["checks in personally"]}, "behavioral_rules": []}'
            ),
        ]
    )
    outcome = await pe.route_and_apply_edit(
        llm, persona=persona, current_difficulty=DifficultyLevel.MEDIUM,
        edit_text="make him more friendly",
    )
    assert outcome.scope == pe.EditScope.PERMANENT
    assert outcome.persona.personality.traits == ["warm", "approachable"]
    assert outcome.persona.version == 2
    assert outcome.new_difficulty is None
    assert len(llm.calls) == 2  # scope classification + edit generation


async def test_route_session_edit_end_to_end():
    persona = make_persona()
    llm = MockLLMClient(responses=["session", "harder"])
    outcome = await pe.route_and_apply_edit(
        llm, persona=persona, current_difficulty=DifficultyLevel.MEDIUM,
        edit_text="make today's conversation harder",
    )
    assert outcome.scope == pe.EditScope.SESSION
    assert outcome.new_difficulty == DifficultyLevel.HARD
    assert outcome.persona is None
    assert len(llm.calls) == 2  # scope classification + shift classification
