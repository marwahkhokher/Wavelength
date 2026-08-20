"""Session Context Builder (Armeen's ownership).

Takes raw scenario descriptions and persona details to produce SessionContext DTOs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

# Ensure voice-tech-infra src is in path for contracts import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-tech-infra" / "src"))

from wavelength_voice.ai_service.contracts import (
    BaselineMetrics,
    PersonaProfile,
    ScenarioConfig,
    SessionContext,
)
from .tone_rules import get_mode_tone_prompt


def build_session_context(
    session_id: str,
    user_id: str,
    mode: Literal["professional", "personal"],
    scenario_title: str,
    scenario_description: str,
    persona_name: str,
    persona_role: str,
    difficulty: Literal["easy", "medium", "hard"] = "medium",
    duration_seconds: int = 300,
    baseline_metrics: BaselineMetrics | None = None,
) -> SessionContext:
    """Builds a structured SessionContext DTO for the session."""
    tone_instructions = get_mode_tone_prompt(mode)
    
    scenario = ScenarioConfig(
        title=scenario_title,
        description=scenario_description,
        setting="Live Session Practice",
    )
    
    persona = PersonaProfile(
        name=persona_name,
        role_description=persona_role,
        communication_style=tone_instructions,
        attitude="firm but receptive" if mode == "professional" else "casual and candid",
        tone_traits=["diplomatic"] if mode == "professional" else ["colloquial"],
    )
    
    return SessionContext(
        session_id=session_id,
        user_id=user_id,
        mode=mode,
        difficulty=difficulty,
        duration_seconds=duration_seconds,
        scenario=scenario,
        persona=persona,
        baseline_metrics=baseline_metrics or BaselineMetrics(),
    )
