"""Finalized session-context handoff to the existing live orchestrator."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from wavelength_voice.ai_service.contracts import BaselineMetrics, SessionContext

from .adapter import build_session_context_from_backend

Orchestrator = TypeVar("Orchestrator")


def handoff_finalized_session(
    session: object,
    orchestrator_factory: Callable[[SessionContext], Orchestrator],
    *,
    scenario_title: str,
    scenario_setting: str,
    baseline_metrics: Mapping[str, Any] | BaselineMetrics | None = None,
) -> Orchestrator:
    """Build a validated context and pass it to an existing orchestrator.

    The adapter performs the finalization check. The factory is injected so
    this function can hand the context to ``VoicePipelineOrchestrator`` or a
    transport-owned equivalent without changing either interface.
    """
    context = build_session_context_from_backend(
        session,
        scenario_title=scenario_title,
        scenario_setting=scenario_setting,
        baseline_metrics=baseline_metrics,
    )
    return orchestrator_factory(context)