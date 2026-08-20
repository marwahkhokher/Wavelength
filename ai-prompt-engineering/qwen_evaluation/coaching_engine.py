"""Coaching Page Engine (Ahmed's ownership).

Generates 'Before vs After' answer refinements and actionable improvement guidance.
"""

from __future__ import annotations

import sys
import re
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
        """Generate evidence-based refinements and a score-driven action plan.

        The method deliberately does not invent achievements or metrics that a
        user did not state. A model-backed coaching pass can replace the small
        text cleanup helper later without changing this report shape.
        """
        refinements: list[dict[str, str | int]] = []
        for idx, (stt, eval_item) in enumerate(zip(transcripts, evaluations)):
            if stt.filler_word_count > 0 or eval_item.scores.structure < 70.0 or eval_item.scores.clarity < 70.0:
                refinements.append(
                    {
                        "turn_index": eval_item.turn_index or idx + 1,
                        "original_user_answer": stt.transcript,
                        "refined_better_answer": self._refine_answer(stt.transcript),
                        "why_it_is_better": self._refinement_reason(stt, eval_item),
                        "key_skill_boosted": self._weakest_skill(eval_item),
                    }
                )

        action_plan = self._action_plan(evaluations)
        return {
            "session_id": session_id,
            "summary_feedback": self._summary(evaluations),
            "answer_refinements": refinements,
            "improvement_action_plan": action_plan,
        }

    @staticmethod
    def _refine_answer(transcript: str) -> str:
        cleaned = re.sub(r"\b(?:um|uh|like|basically|actually)\b,?\s*", "", transcript, flags=re.IGNORECASE)
        cleaned = re.sub(r"\byou know\b,?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,")
        if not cleaned:
            return "State the main point, then support it with one specific example."
        cleaned = cleaned[0].upper() + cleaned[1:]
        if cleaned[-1] not in ".!?":
            cleaned += "."
        return cleaned

    @staticmethod
    def _weakest_skill(evaluation: PerTurnEvaluation) -> str:
        scores = {
            "Clarity": evaluation.scores.clarity,
            "Confidence": evaluation.scores.confidence,
            "Structure": evaluation.scores.structure,
            "Fluency": evaluation.scores.fluency,
            "Filler-word control": evaluation.scores.filler_words_score,
        }
        return min(scores, key=scores.get)  # type: ignore[arg-type]

    def _refinement_reason(self, stt: STTResult, evaluation: PerTurnEvaluation) -> str:
        reasons = []
        if stt.filler_word_count:
            reasons.append("removes filler language")
        if evaluation.scores.structure < 70:
            reasons.append("makes the main point easier to follow")
        if evaluation.scores.clarity < 70:
            reasons.append("uses a more direct opening")
        return "It " + (", ".join(reasons) if reasons else "preserves the original meaning more concisely") + "."

    def _summary(self, evaluations: list[PerTurnEvaluation]) -> str:
        if not evaluations:
            return "Complete a conversation to receive an evidence-based coaching summary."
        averages = {
            "clarity": sum(item.scores.clarity for item in evaluations) / len(evaluations),
            "confidence": sum(item.scores.confidence for item in evaluations) / len(evaluations),
            "structure": sum(item.scores.structure for item in evaluations) / len(evaluations),
            "fluency": sum(item.scores.fluency for item in evaluations) / len(evaluations),
            "filler-word control": sum(item.scores.filler_words_score for item in evaluations) / len(evaluations),
        }
        weakest = min(averages, key=averages.get)
        return f"Your clearest improvement opportunity across this session is {weakest}."

    def _action_plan(self, evaluations: list[PerTurnEvaluation]) -> list[dict[str, str]]:
        if not evaluations:
            return []
        averages = {
            "Structure": sum(item.scores.structure for item in evaluations) / len(evaluations),
            "Confidence": sum(item.scores.confidence for item in evaluations) / len(evaluations),
            "Fluency": sum(item.scores.fluency for item in evaluations) / len(evaluations),
            "Clarity": sum(item.scores.clarity for item in evaluations) / len(evaluations),
            "Filler-word control": sum(item.scores.filler_words_score for item in evaluations) / len(evaluations),
        }
        guidance = {
            "Structure": "Use point, evidence, and result for each answer.",
            "Confidence": "Pause once, then use direct language and a steady pace.",
            "Fluency": "Practice at a comfortable pace and keep pauses brief.",
            "Clarity": "Lead with your conclusion before adding supporting detail.",
            "Filler-word control": "Replace filler words with a silent breath before continuing.",
        }
        weakest_areas = sorted(averages, key=averages.get)[:2]
        return [
            {
                "area": area,
                "current_level": f"{averages[area] / 10:.1f}/10",
                "target_level": f"{min(9.0, averages[area] / 10 + 2):.1f}/10",
                "issue_identified": f"{area} was the lowest-scoring area in this session.",
                "actionable_how_to": guidance[area],
            }
            for area in weakest_areas
        ]
