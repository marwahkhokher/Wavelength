from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-tech-infra" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from session_context.handoff import handoff_finalized_session


def backend_session(**overrides: object) -> dict[str, object]:
    session: dict[str, object] = {
        "id": "session-1",
        "user_id": "user-1",
        "mode": "professional",
        "scenario_text": "Discuss a promotion after a strong performance year.",
        "persona_profile": {
            "name": "Alex",
            "role_or_relationship": "Direct manager",
            "speech_style": "Measured and concise",
            "personality_traits": ["direct"],
        },
        "persona_finalized": True,
        "difficulty": "hard",
        "duration_seconds": 300,
    }
    session.update(overrides)
    return session


def test_finalized_context_reaches_existing_orchestrator_factory() -> None:
    received = []

    def orchestrator_factory(context):  # noqa: ANN001
        received.append(context)
        return context

    result = handoff_finalized_session(
        backend_session(),
        orchestrator_factory,
        scenario_title="Promotion discussion",
        scenario_setting="Annual performance review",
    )

    assert result is received[0]
    assert result.session_id == "session-1"
    assert result.mode == "professional"
    assert result.difficulty == "hard"
    assert result.duration_seconds == 300
    assert result.scenario.description.startswith("Discuss a promotion")
    assert result.persona.name == "Alex"
    assert result.persona.role_description == "Direct manager"


def test_personal_finalized_context_reaches_orchestrator() -> None:
    received = []

    def orchestrator_factory(context):  # noqa: ANN001
        received.append(context)
        return context

    handoff_finalized_session(
        backend_session(
            mode="personal",
            difficulty="medium",
            persona_profile={
                "name": "Sam",
                "role_or_relationship": "Friend",
                "speech_style": "Direct and casual",
            },
        ),
        orchestrator_factory,
        scenario_title="Cancelled plans",
        scenario_setting="Personal conversation",
    )

    assert received[0].mode == "personal"
    assert received[0].difficulty == "medium"
    assert received[0].persona.communication_style == "Direct and casual"


def test_non_finalized_session_cannot_reach_orchestrator() -> None:
    called = False

    def orchestrator_factory(context):  # noqa: ANN001
        nonlocal called
        called = True
        return context

    with pytest.raises(ValueError, match="must be finalized"):
        handoff_finalized_session(
            backend_session(persona_finalized=False),
            orchestrator_factory,
            scenario_title="Promotion discussion",
            scenario_setting="Annual performance review",
        )

    assert called is False