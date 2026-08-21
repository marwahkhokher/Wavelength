"""Adapt finalized backend session data to the existing voice contract."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Keep this module usable from the AI prompt-engineering package as it exists today.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-tech-infra" / "src"))

from wavelength_voice.ai_service.contracts import (  # noqa: E402
    BaselineMetrics,
    PersonaProfile,
    ScenarioConfig,
    SessionContext,
)

VALID_MODES = {"professional", "personal"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def build_session_context_from_backend(
    session: object,
    *,
    scenario_title: str | None = None,
    scenario_setting: str | None = None,
    baseline_metrics: Mapping[str, Any] | BaselineMetrics | None = None,
) -> SessionContext:
    """Convert a finalized backend session into the existing SessionContext.

    ``session`` may be a SQLAlchemy ConversationSession or a mapping with the
    same persisted field names. The backend currently stores persona data as
    JSON, so this adapter deliberately validates the minimum fields required
    by the voice-infra contract at this boundary.
    """
    if not _read(session, "persona_finalized"):
        raise ValueError("Persona must be finalized before building SessionContext")

    session_id = _required(session, "id", "session_id")
    user_id = _required(session, "user_id")
    mode = _enum_value(_required(session, "mode"))
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported mode: {mode!r}")

    difficulty = _enum_value(_required(session, "difficulty"))
    if difficulty not in VALID_DIFFICULTIES:
        raise ValueError(f"Unsupported difficulty: {difficulty!r}")

    scenario_text = _required(session, "scenario_text")
    title = scenario_title or _read(session, "scenario_title")
    setting = scenario_setting or _read(session, "scenario_setting")
    if not title or not setting:
        raise ValueError(
            "Scenario title and setting are required by the existing "
            "SessionContext contract; the backend currently stores only scenario_text"
        )

    persona_data = _read(session, "persona_profile")
    if not isinstance(persona_data, Mapping) or not persona_data:
        raise ValueError("Finalized session must contain a persona_profile mapping")

    persona_name = _required_mapping(persona_data, "name")
    role_description = persona_data.get("role_description") or persona_data.get("role_or_relationship")
    if not role_description:
        raise ValueError("persona_profile requires role_description or role_or_relationship")

    communication_style = persona_data.get("communication_style") or persona_data.get("speech_style")
    if not communication_style:
        raise ValueError(
            "persona_profile requires communication_style or speech_style"
        )
    if isinstance(communication_style, list):
        communication_style = "; ".join(str(item) for item in communication_style)

    tone_traits = persona_data.get("tone_traits") or []
    if isinstance(tone_traits, str):
        tone_traits = [tone_traits]

    metrics = baseline_metrics
    if metrics is None:
        metrics = _read(session, "baseline_metrics")
    if isinstance(metrics, Mapping):
        metrics = BaselineMetrics(**metrics)

    duration_seconds = _required(session, "duration_seconds")

    # These are the only persona fields represented by the existing contract.
    # Rich fields such as age_range, background, goals, triggers, tone details,
    # initial_state, provenance, and is_finalized remain outside this DTO.
    persona = PersonaProfile(
        name=str(persona_name),
        role_description=str(role_description),
        communication_style=str(communication_style),
        attitude=str(persona_data.get("attitude", "neutral")),
        tone_traits=list(tone_traits),
    )

    return SessionContext(
        session_id=str(session_id),
        user_id=str(user_id),
        mode=mode,
        difficulty=difficulty,
        duration_seconds=duration_seconds,
        scenario=ScenarioConfig(
            title=str(title),
            description=str(scenario_text),
            setting=str(setting),
        ),
        persona=persona,
        baseline_metrics=metrics if isinstance(metrics, BaselineMetrics) else BaselineMetrics(),
    )


def _read(source: object, field: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(field)
    return getattr(source, field, None)


def _required(source: object, *fields: str) -> Any:
    for field in fields:
        value = _read(source, field)
        if value is not None and value != "":
            return value
    raise ValueError(f"Missing required session field: {fields[0]}")


def _required_mapping(source: Mapping[str, Any], field: str) -> Any:
    value = source.get(field)
    if value is None or value == "":
        raise ValueError(f"Missing required persona field: {field}")
    return value


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value))