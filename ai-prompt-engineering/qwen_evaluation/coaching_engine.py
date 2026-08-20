"""Coaching Page Engine (Ahmed's ownership).

Generates 'Before vs After' answer refinements and actionable improvement guidance.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-tech-infra" / "src"))

from wavelength_voice.ai_service.contracts import PerTurnEvaluation, STTResult


class CoachingEngine:
    """LLM Coaching Engine for post-session answer refinement and feedback."""

    def generate_coaching_report(
        self,
        session_id: str,
        transcripts: list[STTResult],
        evaluations: list[PerTurnEvaluation],
    ) -> dict[str, Any]:
        """Generates answer refinements ('Before vs After') and actionable how-to tips."""
        refinements = []
        for idx, (stt, eval_item) in enumerate(zip(transcripts, evaluations)):
            if stt.filler_word_count > 0 or eval_item.scores.structure < 70.0:
                refinements.append(
                    {
                        "turn_index": idx + 1,
                        "original_user_answer": stt.transcript,
                        "refined_better_answer": (
                            "Over the last quarter, my work on pipeline refactoring "
                            "directly increased team velocity by 25%."
                        ),
                        "why_it_is_better": (
                            "Eliminates filler words ('um', 'like') and adds quantitative metrics."
                        ),
                        "key_skill_boosted": "Clarity & Confidence",
                    }
                )
                
        return {
            "session_id": session_id,
            "summary_feedback": "Strong domain relevance, but structure dropped during objection handling.",
            "answer_refinements": refinements,
            "improvement_action_plan": [
                {
                    "area": "Structure",
                    "issue_identified": "Hesitated when manager pushed back on compensation.",
                    "actionable_how_to": (
                        "Use the STAR method (Situation, Task, Action, Result) before answering."
                    ),
                }
            ],
        }
