"""Conversation Prompt LLM (Taha's ownership).

Consumes Qwen's deep evaluation report, STT transcript, tone result, persona, and history to generate the next response.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-tech-infra" / "src"))

from wavelength_voice.ai_service.contracts import PerTurnEvaluation, PromptLLMInput, PromptLLMOutput


class ConversationLLM:
    """Prompt Generation LLM executing Conversation Roleplay Mode."""

    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name

    async def generate_next_turn(
        self,
        payload: PromptLLMInput,
        qwen_eval: PerTurnEvaluation,
    ) -> PromptLLMOutput:
        """Generates the next conversation turn using Qwen's deep answer evaluation."""
        mode = payload.session_context.mode
        persona_name = payload.session_context.persona.name
        
        # Tone register selection based on Professional vs Personal mode (Section 6 PRD)
        if mode == "professional":
            reply = (
                f"That is an interesting point regarding your team velocity. "
                f"However, as {persona_name}, could you clarify what specific metrics supported that growth?"
            )
        else:
            reply = (
                f"Yaar, that sounds okay, but how do I know the velocity actually went up? "
                f"Do you have the numbers for that?"
            )
            
        return PromptLLMOutput(
            reply_text=reply,
            end_session=False,
            persona_state_update=f"Evaluated score: {qwen_eval.scores.overall_turn_score}",
        )
