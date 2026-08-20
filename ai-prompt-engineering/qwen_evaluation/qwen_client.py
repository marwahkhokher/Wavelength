"""Managed Qwen client using Alibaba Model Studio's OpenAI-compatible API."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-tech-infra" / "src"))

from wavelength_voice.ai_service.contracts import PerTurnEvaluation

DEFAULT_MODEL = "qwen-plus"


class LiveQwenEvaluationClient:
    """Converts a structured turn payload into a validated Qwen evaluation."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model or os.getenv("QWEN_MODEL", DEFAULT_MODEL)
        if client is not None:
            self.client = client
            return

        resolved_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        resolved_base_url = base_url or os.getenv("QWEN_BASE_URL")
        if not resolved_key or not resolved_base_url:
            raise ValueError("DASHSCOPE_API_KEY and QWEN_BASE_URL are required for live Qwen evaluation.")

        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=resolved_key, base_url=resolved_base_url)

    async def evaluate(self, payload: dict[str, Any]) -> PerTurnEvaluation:
        completion = await self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise communications evaluator for a spoken conversation. "
                        "Return JSON only, with no Markdown. It must validate as PerTurnEvaluation: "
                        "turn_index, scores (0-100 clarity, empathy, filler_words_score, "
                        "structure, relevance, confidence, fluency, overall_turn_score), "
                        "strengths, areas_for_improvement, coach_tip, main_point_detected, "
                        "structural_assessment, emotional_alignment, key_flaw, and "
                        "suggested_conversation_followup_direction. Base conclusions only on the input."
                    ),
                },
                {"role": "user", "content": json.dumps(payload)},
            ],
        )
        content = completion.choices[0].message.content or ""
        return PerTurnEvaluation.model_validate(self._parse_json(content))

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        """Accept JSON wrapped in an accidental Markdown fence, but nothing else."""
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
            cleaned = cleaned.rsplit("```", 1)[0].strip()
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise ValueError("Qwen returned a JSON value other than an object.")
        return parsed
