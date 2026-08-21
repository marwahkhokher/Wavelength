from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-tech-infra" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from session_context.adapter import build_session_context_from_backend


def backend_session(**overrides: object) -> dict[str, object]:
    session: dict[str, object] = {
        "id": "session-1",
        "user_id": "user-1",
        "mode": "professional",
        "scenario_text": "Discuss a promotion after a strong performance year.",
        "scenario_title": "Promotion discussion",
        "scenario_setting": "Annual performance review",
        "persona_profile": {
            "name": "Alex",
            "role_or_relationship": "Direct manager",
            "speech_style": "Measured and concise",
            "personality_traits": ["direct"],
            "background": "Leads the engineering team.",
            "goals": ["Protect the budget"],
        },
        "persona_finalized": True,
        "difficulty": "medium",
        "duration_seconds": 300,
    }
    session.update(overrides)
    return session


def test_adapts_finalized_professional_session() -> None:
    context = build_session_context_from_backend(backend_session())

    assert context.session_id == "session-1"
    assert context.user_id == "user-1"
    assert context.mode == "professional"
    assert context.scenario.title == "Promotion discussion"
    assert context.scenario.description.startswith("Discuss a promotion")
    assert context.scenario.setting == "Annual performance review"
    assert context.persona.name == "Alex"
    assert context.persona.role_description == "Direct manager"
    assert context.persona.communication_style == "Measured and concise"
    assert context.difficulty == "medium"
    assert context.duration_seconds == 300


def test_rejects_non_finalized_persona() -> None:
    with pytest.raises(ValueError, match="must be finalized"):
        build_session_context_from_backend(backend_session(persona_finalized=False))


def test_adapts_personal_mode() -> None:
    context = build_session_context_from_backend(
        backend_session(
            mode="personal",
            persona_profile={
                "name": "Sam",
                "role_or_relationship": "Sibling",
                "speech_style": "Direct and casual",
            },
        )
    )

    assert context.mode == "personal"
    assert context.persona.communication_style == "Direct and casual"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_accepts_existing_difficulty_values(difficulty: str) -> None:
    context = build_session_context_from_backend(backend_session(difficulty=difficulty))

    assert context.difficulty == difficulty


def test_rejects_invalid_difficulty() -> None:
    with pytest.raises(ValueError, match="Unsupported difficulty"):
        build_session_context_from_backend(backend_session(difficulty="extreme"))


@pytest.mark.parametrize("field", ["id", "user_id", "mode", "scenario_text", "difficulty", "duration_seconds"])
def test_rejects_missing_required_session_input(field: str) -> None:
    session = backend_session()
    session[field] = None

    with pytest.raises(ValueError, match="Missing required session field"):
        build_session_context_from_backend(session)


def test_rejects_missing_required_scenario_contract_fields() -> None:
    with pytest.raises(ValueError, match="Scenario title and setting"):
        build_session_context_from_backend(
            backend_session(scenario_title=None, scenario_setting=None)
        )