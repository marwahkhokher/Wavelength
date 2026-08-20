"""Abstract provider interfaces for teammate integration (Taha's ownership).

Each provider wraps the output of one teammate's component. When the real
implementation is unavailable, a MockProvider can be substituted with zero
changes to the orchestrator.

Architecture:
    TranscriptProvider   ← Person 2 (Areej)  → STTResult
    ToneProvider         ← Person 3 (Zaid)   → ToneResult
    AnalysisProvider     ← Person 4 (Ahmed)  → PerTurnEvaluation
    SessionContextProvider ← Person 1 (Armeen) → SessionContext
"""

from __future__ import annotations

import abc
import sys
from pathlib import Path

# Ensure contracts are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "voice-tech-infra" / "src"))

from wavelength_voice.ai_service.contracts import (
    PerTurnEvaluation,
    SessionContext,
    STTResult,
    ToneResult,
)


class TranscriptProvider(abc.ABC):
    """Interface for obtaining a transcript from user audio (Person 2)."""

    @abc.abstractmethod
    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> STTResult:
        """Transcribe user audio into an STTResult."""
        raise NotImplementedError


class ToneProvider(abc.ABC):
    """Interface for tone/emotion analysis from user audio (Person 3)."""

    @abc.abstractmethod
    async def analyze_tone(
        self,
        audio_bytes: bytes,
        word_count: int,
        duration_sec: float,
    ) -> ToneResult:
        """Analyze the emotional tone of user audio."""
        raise NotImplementedError


class AnalysisProvider(abc.ABC):
    """Interface for deep per-turn evaluation (Person 4)."""

    @abc.abstractmethod
    async def evaluate_turn(
        self,
        turn_index: int,
        stt_result: STTResult,
        tone_result: ToneResult,
    ) -> PerTurnEvaluation:
        """Evaluate the user's turn across multiple dimensions."""
        raise NotImplementedError


class SessionContextProvider(abc.ABC):
    """Interface for building session context (Person 1)."""

    @abc.abstractmethod
    def build_context(
        self,
        session_id: str,
        user_id: str,
        mode: str,
        scenario_title: str,
        scenario_description: str,
        persona_name: str,
        persona_role: str,
        difficulty: str = "medium",
        duration_seconds: int = 300,
    ) -> SessionContext:
        """Build a SessionContext DTO."""
        raise NotImplementedError
