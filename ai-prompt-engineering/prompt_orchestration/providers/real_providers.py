"""Real provider adapters wrapping teammates' actual implementations (Taha's ownership).

These adapters wrap the existing code pushed by Persons 1–4 so the
orchestrator consumes a uniform provider interface regardless of whether
the underlying component is real or mock.

IMPORTANT: DO NOT modify teammates' code. Only wrap it.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure both project roots are importable
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root.parent / "voice-tech-infra" / "src"))

from wavelength_voice.ai_service.contracts import (
    PerTurnEvaluation,
    SessionContext,
    STTResult,
    ToneResult,
)

from .base import AnalysisProvider, SessionContextProvider, ToneProvider, TranscriptProvider


class WhisperTranscriptProvider(TranscriptProvider):
    """Wraps Person 2's (Areej) WhisperSTTEngine."""

    def __init__(self) -> None:
        from stt_filler.stt_engine import WhisperSTTEngine
        self._engine = WhisperSTTEngine()

    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> STTResult:
        return await self._engine.transcribe_audio_bytes(audio_bytes, sample_rate)


class RealToneProvider(ToneProvider):
    """Wraps Person 3's (Zaid) EmotionClassifier."""

    def __init__(self) -> None:
        from tone_analysis.emotion_classifier import EmotionClassifier
        self._classifier = EmotionClassifier()

    async def analyze_tone(
        self,
        audio_bytes: bytes,
        word_count: int,
        duration_sec: float,
    ) -> ToneResult:
        # EmotionClassifier.classify_audio_tone is sync; wrap it
        return self._classifier.classify_audio_tone(audio_bytes, word_count, duration_sec)


class QwenAnalysisProvider(AnalysisProvider):
    """Wraps Person 4's (Ahmed) QwenDeepEvaluator."""

    def __init__(self) -> None:
        from qwen_evaluation.deep_evaluator import QwenDeepEvaluator
        self._evaluator = QwenDeepEvaluator()

    async def evaluate_turn(
        self,
        turn_index: int,
        stt_result: STTResult,
        tone_result: ToneResult,
    ) -> PerTurnEvaluation:
        return await self._evaluator.evaluate_turn(turn_index, stt_result, tone_result)


class RealSessionContextProvider(SessionContextProvider):
    """Wraps Person 1's (Armeen) build_session_context."""

    def build_context(
        self,
        session_id: str,
        user_id: str,
        mode: str = "professional",
        scenario_title: str = "Salary Negotiation",
        scenario_description: str = "Negotiating a raise with your manager.",
        persona_name: str = "David Miller",
        persona_role: str = "VP of Engineering",
        difficulty: str = "medium",
        duration_seconds: int = 300,
    ) -> SessionContext:
        from session_context.builder import build_session_context

        return build_session_context(
            session_id=session_id,
            user_id=user_id,
            mode=mode,  # type: ignore[arg-type]
            scenario_title=scenario_title,
            scenario_description=scenario_description,
            persona_name=persona_name,
            persona_role=persona_role,
            difficulty=difficulty,  # type: ignore[arg-type]
            duration_seconds=duration_seconds,
        )
