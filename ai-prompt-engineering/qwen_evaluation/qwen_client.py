"""Qwen client for any OpenAI-compatible endpoint - a local server (e.g. Ollama)
by default, or Alibaba Model Studio's hosted API if configured.

Called once per session, after the conversation ends, rather than per turn:
a single call on CPU-only local hardware can take several minutes, which is
too slow to sit in the live turn-taking loop. See
QwenDeepEvaluator.evaluate_session.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-tech-infra" / "src"))

from wavelength_voice.ai_service.contracts import PerTurnEvaluation

DEFAULT_MODEL = "qwen3.5:4b"
# Local OpenAI-compatible servers (Ollama, vLLM, LM Studio) don't check the API
# key, but the OpenAI SDK requires a non-empty string to be passed.
LOCAL_PLACEHOLDER_API_KEY = "not-needed-for-local-server"

SESSION_SYSTEM_PROMPT = (
    "You are a precise communications evaluator reviewing an entire practice "
    "conversation, not a single turn. You are given every user turn from the "
    "session, each with its transcript and tone data, plus the scenario "
    "context. Return JSON only, with no Markdown, as ONE holistic evaluation "
    "of the full session. It must validate as PerTurnEvaluation: turn_index "
    "(set to the total number of turns), scores (0-100 clarity, empathy, "
    "filler_words_score, structure, relevance, confidence, fluency, "
    "overall_turn_score) reflecting the session as a whole, strengths, "
    "areas_for_improvement, coach_tip, main_point_detected (the overall "
    "theme across turns), structural_assessment, emotional_alignment (how "
    "tone evolved across the session), key_flaw, and "
    "suggested_conversation_followup_direction (what to practice next). "
    "Base conclusions only on the input."
)


class LiveQwenEvaluationClient:
    """Runs one holistic Qwen review of a full conversation session."""

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

        resolved_base_url = base_url or os.getenv("QWEN_BASE_URL")
        if not resolved_base_url:
            raise ValueError("QWEN_BASE_URL is required for live Qwen evaluation.")
        resolved_key = (
            api_key
            or os.getenv("QWEN_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
            or LOCAL_PLACEHOLDER_API_KEY
        )

        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=resolved_key, base_url=resolved_base_url)

    async def evaluate_session(self, payload: dict[str, Any]) -> PerTurnEvaluation:
        completion = await self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": SESSION_SYSTEM_PROMPT},
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
