"""Qwen Deep Evaluator (Ahmed's ownership).

Performs per-turn evaluation on Clarity, Empathy, Filler score, Structure, Relevance, Fluency.
Output is passed to Prompt Generation LLM per turn.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-tech-infra" / "src"))

from wavelength_voice.ai_service.contracts import MetricScores, PerTurnEvaluation, STTResult, ToneResult


class QwenDeepEvaluator:
    """Qwen model wrapper for per-turn deep evaluation."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-72B-Instruct"):
        self.model_name = model_name

    async def evaluate_turn(
        self,
        turn_index: int,
        stt_result: STTResult,
        tone_result: ToneResult,
    ) -> PerTurnEvaluation:
        """Deeply evaluates user utterance against PRD metrics."""
        # Simulated Qwen deep evaluation output
        scores = MetricScores(
            clarity=78.0,
            empathy=65.0,
            filler_words_score=70.0,
            structure=65.0,
            relevance=90.0,
            confidence=60.0,
            overall_turn_score=71.3,
        )
        
        return PerTurnEvaluation(
            turn_index=turn_index,
            scores=scores,
            strengths=["High relevance to prompt scenario"],
            areas_for_improvement=["Initial hesitation and filler word usage"],
            coach_tip="Pause 1 second before speaking to improve structure.",
        )
