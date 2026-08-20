"""Qwen Deep Evaluator (Ahmed's ownership).

Performs per-turn evaluation on Clarity, Empathy, Filler score, Structure, Relevance, Fluency.
Output is passed to Prompt Generation LLM per turn.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-tech-infra" / "src"))

from wavelength_voice.ai_service.contracts import (
    MetricScores,
    PerTurnEvaluation,
    SessionContext,
    STTResult,
    ToneResult,
)

from .qwen_client import LiveQwenEvaluationClient

logger = logging.getLogger(__name__)


class EvaluationClient(Protocol):
    async def evaluate(self, payload: dict[str, Any]) -> PerTurnEvaluation: ...


class QwenDeepEvaluator:
    """Qwen model wrapper for per-turn deep evaluation."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-72B-Instruct",
        live_client: EvaluationClient | None = None,
    ):
        self.model_name = model_name
        self.live_client = live_client
        if self.live_client is None and os.getenv("DASHSCOPE_API_KEY") and os.getenv("QWEN_BASE_URL"):
            self.live_client = LiveQwenEvaluationClient(model=os.getenv("QWEN_MODEL"))
        elif self.live_client is None and os.getenv("DASHSCOPE_API_KEY"):
            logger.warning("DASHSCOPE_API_KEY is set but QWEN_BASE_URL is missing; using deterministic evaluation.")

    async def evaluate_turn(
        self,
        turn_index: int,
        stt_result: STTResult,
        tone_result: ToneResult,
        session_context: SessionContext | None = None,
    ) -> PerTurnEvaluation:
        """Evaluate one turn from the STT and acoustic contracts.

        This local scorer is intentionally deterministic: it keeps live turns
        usable without a hosted Qwen credential and creates an auditable input
        baseline for a future model-backed evaluator.
        """
        if self.live_client is not None:
            try:
                return await self.live_client.evaluate(
                    self._live_payload(turn_index, stt_result, tone_result, session_context)
                )
            except Exception:  # Keep live practice usable during provider outages or invalid JSON.
                logger.exception("Live Qwen evaluation failed; using deterministic fallback.")

        return self._evaluate_deterministically(turn_index, stt_result, tone_result, session_context)

    def _evaluate_deterministically(
        self,
        turn_index: int,
        stt_result: STTResult,
        tone_result: ToneResult,
        session_context: SessionContext | None,
    ) -> PerTurnEvaluation:
        transcript = stt_result.transcript.strip()
        words = [word.strip(".,!?;:").lower() for word in transcript.split()]
        word_count = max(stt_result.total_words, len(words), 1)
        filler_ratio = stt_result.filler_word_count / word_count

        clarity = self._clamp(50 + min(word_count, 24) * 2 - filler_ratio * 80)
        filler_words_score = self._clamp(100 - filler_ratio * 300)
        structure = self._structure_score(transcript, words, filler_ratio)
        fluency = self._fluency_score(tone_result, filler_ratio)
        confidence = self._confidence_score(tone_result)
        relevance = self._relevance_score(words, session_context)
        empathy = self._empathy_score(words)
        scores = MetricScores(
            clarity=round(clarity, 1),
            empathy=round(empathy, 1),
            filler_words_score=round(filler_words_score, 1),
            structure=round(structure, 1),
            relevance=round(relevance, 1),
            confidence=round(confidence, 1),
            fluency=round(fluency, 1),
            overall_turn_score=round(
                clarity * 0.20
                + confidence * 0.20
                + structure * 0.20
                + relevance * 0.15
                + fluency * 0.15
                + filler_words_score * 0.10,
                1,
            ),
        )

        main_point = self._main_point(transcript)
        strengths = self._strengths(scores)
        improvements = self._improvements(scores)
        key_flaw = improvements[0] if improvements else "No major delivery issue detected."

        return PerTurnEvaluation(
            turn_index=turn_index,
            scores=scores,
            strengths=strengths,
            areas_for_improvement=improvements,
            coach_tip=self._coach_tip(scores),
            main_point_detected=main_point,
            structural_assessment=self._structural_assessment(scores),
            emotional_alignment=(
                f"Tone was classified as {tone_result.primary_emotion} "
                f"with a hesitation score of {tone_result.hesitation_score:.2f}."
            ),
            key_flaw=key_flaw,
            suggested_conversation_followup_direction=self._followup_direction(scores, main_point),
        )

    @staticmethod
    def _live_payload(
        turn_index: int,
        stt_result: STTResult,
        tone_result: ToneResult,
        session_context: SessionContext | None,
    ) -> dict[str, Any]:
        return {
            "turn_index": turn_index,
            "stt_result": stt_result.model_dump(mode="json"),
            "tone_result": tone_result.model_dump(mode="json"),
            "session_context": session_context.model_dump(mode="json") if session_context else None,
        }

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(100.0, value))

    def _structure_score(self, transcript: str, words: list[str], filler_ratio: float) -> float:
        signposts = {"because", "therefore", "first", "then", "finally", "result", "for", "example"}
        evidence = sum(word in signposts for word in words)
        sentence_bonus = 10 if transcript.endswith((".", "!", "?")) else 0
        return self._clamp(45 + evidence * 9 + sentence_bonus - filler_ratio * 60)

    def _fluency_score(self, tone_result: ToneResult, filler_ratio: float) -> float:
        pace_distance = abs(tone_result.speech_rate_wpm - 140)
        pace_score = max(0.0, 100 - pace_distance * 0.6)
        pause_penalty = tone_result.silence_ratio * 35 + min(tone_result.pause_metrics.max_pause_sec, 3) * 5
        return self._clamp(pace_score - pause_penalty - filler_ratio * 40)

    def _confidence_score(self, tone_result: ToneResult) -> float:
        emotion_adjustments = {
            "confident": 22,
            "energetic": 12,
            "empathetic": 6,
            "neutral": 0,
            "monotone": -8,
            "hesitant": -22,
            "anxious": -25,
            "frustrated": -15,
        }
        return self._clamp(72 + emotion_adjustments[tone_result.primary_emotion] - tone_result.hesitation_score * 25)

    def _relevance_score(self, words: list[str], session_context: SessionContext | None) -> float:
        if session_context is None:
            return self._clamp(55 + min(len(words), 20) * 2)
        scenario_words = {
            word.strip(".,!?;:").lower()
            for word in f"{session_context.scenario.title} {session_context.scenario.description}".split()
            if len(word.strip(".,!?;:")) > 3
        }
        overlap = len(set(words) & scenario_words)
        return self._clamp(55 + min(len(words), 18) * 1.5 + overlap * 8)

    def _empathy_score(self, words: list[str]) -> float:
        empathy_markers = {"understand", "appreciate", "thank", "we", "together", "support", "help"}
        return self._clamp(58 + sum(word in empathy_markers for word in words) * 10)

    @staticmethod
    def _main_point(transcript: str) -> str:
        return transcript if len(transcript) <= 180 else f"{transcript[:177].rstrip()}..."

    @staticmethod
    def _strengths(scores: MetricScores) -> list[str]:
        strengths = []
        for label, value in (("Relevance", scores.relevance), ("Clarity", scores.clarity), ("Fluency", scores.fluency)):
            if value >= 75:
                strengths.append(f"{label} was strong on this turn.")
        return strengths or ["The response contained a usable main point."]

    @staticmethod
    def _improvements(scores: MetricScores) -> list[str]:
        candidates = [
            (scores.filler_words_score, "Reduce filler words and begin with the main point."),
            (scores.confidence, "Use a steadier pace and more decisive language."),
            (scores.structure, "Use a clear point, evidence, and result structure."),
            (scores.fluency, "Aim for a comfortable pace with shorter pauses."),
        ]
        return [message for value, message in candidates if value < 70]

    @staticmethod
    def _coach_tip(scores: MetricScores) -> str:
        if scores.filler_words_score < 70:
            return "Pause silently before answering instead of filling the gap with a filler word."
        if scores.structure < 70:
            return "State your point first, then support it with one concrete example."
        if scores.confidence < 70:
            return "Slow down slightly and end each key point with a clear conclusion."
        return "Keep using concrete evidence to make the next response even stronger."

    @staticmethod
    def _structural_assessment(scores: MetricScores) -> str:
        if scores.structure >= 75:
            return "The response had a clear, easy-to-follow shape."
        return "The response would benefit from an explicit point, evidence, and result sequence."

    @staticmethod
    def _followup_direction(scores: MetricScores, main_point: str) -> str:
        if scores.relevance < 70:
            return "Please connect your answer directly to this scenario."
        if scores.structure < 70:
            return "Could you give one specific example that supports your main point?"
        if scores.filler_words_score < 70:
            return "Please restate your main point concisely, without filler words."
        return f"Could you share measurable evidence supporting: {main_point}"
