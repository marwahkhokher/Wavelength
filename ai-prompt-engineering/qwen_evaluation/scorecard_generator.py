"""Session Scorecard Generator (Ahmed's ownership).

Aggregates per-turn evaluations into final 1–10 metrics and compares against onboarding baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-tech-infra" / "src"))

from wavelength_voice.ai_service.contracts import (
    BaselineMetrics,
    FinalSessionEvaluation,
    MetricScores,
    PerTurnEvaluation,
)


def generate_session_scorecard(
    session_id: str,
    turn_evaluations: list[PerTurnEvaluation],
    baseline: BaselineMetrics,
) -> FinalSessionEvaluation:
    """Generates the final 1–10 metrics scorecard and calculates improvement deltas."""
    if not turn_evaluations:
        avg_scores = MetricScores(
            clarity=50.0,
            empathy=50.0,
            filler_words_score=50.0,
            structure=50.0,
            relevance=50.0,
            confidence=50.0,
            fluency=50.0,
            overall_turn_score=50.0,
        )
    else:
        n = len(turn_evaluations)
        avg_scores = MetricScores(
            clarity=round(sum(e.scores.clarity for e in turn_evaluations) / n, 1),
            empathy=round(sum(e.scores.empathy for e in turn_evaluations) / n, 1),
            filler_words_score=round(sum(e.scores.filler_words_score for e in turn_evaluations) / n, 1),
            structure=round(sum(e.scores.structure for e in turn_evaluations) / n, 1),
            relevance=round(sum(e.scores.relevance for e in turn_evaluations) / n, 1),
            confidence=round(sum(e.scores.confidence for e in turn_evaluations) / n, 1),
            fluency=round(sum(e.scores.fluency for e in turn_evaluations) / n, 1),
            overall_turn_score=round(sum(e.scores.overall_turn_score for e in turn_evaluations) / n, 1),
        )
        
    deltas = {
        "clarity_delta": round((avg_scores.clarity / 10.0) - baseline.clarity, 1),
        "confidence_delta": round((avg_scores.confidence / 10.0) - baseline.confidence, 1),
        "structure_delta": round((avg_scores.structure / 10.0) - baseline.structure, 1),
        "relevance_delta": round((avg_scores.relevance / 10.0) - baseline.relevance, 1),
        "fluency_delta": round((avg_scores.fluency / 10.0) - baseline.fluency, 1),
        "filler_words_delta": round((avg_scores.filler_words_score / 10.0) - baseline.filler_words_score, 1),
    }
    
    summary = [
        f"Clarity: {avg_scores.clarity / 10:.1f}/10",
        f"Confidence: {avg_scores.confidence / 10:.1f}/10",
        f"Structure: {avg_scores.structure / 10:.1f}/10",
        f"Relevance: {avg_scores.relevance / 10:.1f}/10",
        f"Fluency: {avg_scores.fluency / 10:.1f}/10",
        f"Filler words: {avg_scores.filler_words_score / 10:.1f}/10",
    ]
    
    return FinalSessionEvaluation(
        session_id=session_id,
        total_turns=len(turn_evaluations),
        average_scores=avg_scores,
        score_deltas_from_baseline=deltas,
        coach_mode_summary=summary,
    )
