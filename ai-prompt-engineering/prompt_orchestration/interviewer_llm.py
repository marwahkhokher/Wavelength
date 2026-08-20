"""Interviewer Prompt LLM (Taha's ownership).

Consumes Qwen's deep evaluation report, STT transcript, tone result, persona,
and conversation history to generate the next persona-consistent response.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Add project roots to path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root.parent / "voice-tech-infra" / "src"))

from wavelength_voice.ai_service.contracts import (
    PerTurnEvaluation,
    PromptLLMInput,
    PromptLLMOutput,
)

from .conversation_memory import ConversationMemory
from .llm_client import BaseLLMClient, get_llm_client
from .models import ConversationState
from .prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class InterviewerLLM:
    """Prompt Generation LLM executing Interviewer / Persona Roleplay Mode."""

    def __init__(
        self,
        llm_client: BaseLLMClient | None = None,
        memory_manager: ConversationMemory | None = None,
    ) -> None:
        self.llm_client = llm_client or get_llm_client()
        self.memory_manager = memory_manager or ConversationMemory()

    async def generate_next_turn(
        self,
        payload: PromptLLMInput,
        qwen_eval: PerTurnEvaluation | None = None,
        conversation_state: ConversationState | None = None,
    ) -> PromptLLMOutput:
        """Generates the next interviewer turn using full context and evaluations."""
        t0 = time.perf_counter()

        prompt_builder = PromptBuilder(
            session_context=payload.session_context,
            memory_manager=self.memory_manager,
        )

        # Create or adapt internal state
        if conversation_state is None:
            conversation_state = ConversationState(
                session_id=payload.session_context.session_id,
                user_id=payload.session_context.user_id,
            )
            # Rehydrate from history if needed
            for turn in payload.conversation_history:
                conversation_state.add_message("user", turn.user_transcript)
                conversation_state.add_message("assistant", turn.ai_response)

        # Update dynamic persona state if evaluation is provided
        if qwen_eval:
            prompt_builder.persona_engine.update_persona_state(
                persona_state=conversation_state.persona_state,
                confidence_score=qwen_eval.scores.confidence,
                clarity_score=qwen_eval.scores.clarity,
                relevance_score=qwen_eval.scores.relevance,
            )

        # Build prompt
        system_instruction = prompt_builder.build_system_prompt(conversation_state)
        turn_prompt = prompt_builder.build_turn_prompt(
            state=conversation_state,
            latest_user_text=payload.current_stt.transcript,
            qwen_eval=qwen_eval,
        )

        # Call LLM
        reply_text = await self.llm_client.generate_response(
            system_instruction=system_instruction,
            prompt=turn_prompt,
        )

        # Update structured memory with facts
        self.memory_manager.extract_facts(
            state=conversation_state,
            user_message=payload.current_stt.transcript,
            ai_response=reply_text,
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "InterviewerLLM turn generated in %.1fms for session %s",
            latency_ms,
            payload.session_context.session_id,
        )

        persona_update = (
            f"Disposition: rec={conversation_state.persona_state.receptiveness:.2f}, "
            f"sat={conversation_state.persona_state.satisfaction:.2f}"
        )

        return PromptLLMOutput(
            reply_text=reply_text,
            end_session=False,
            persona_state_update=persona_update,
        )

    async def generate_opening(
        self,
        session_context,
        conversation_state: ConversationState | None = None,
    ) -> str:
        """Generates the opening scenario greeting/question from the persona."""
        prompt_builder = PromptBuilder(
            session_context=session_context,
            memory_manager=self.memory_manager,
        )
        if conversation_state is None:
            conversation_state = ConversationState(
                session_id=session_context.session_id,
                user_id=session_context.user_id,
            )

        system_instruction = prompt_builder.build_system_prompt(conversation_state)
        opening_prompt = prompt_builder.build_opening_prompt(conversation_state)

        return await self.llm_client.generate_response(
            system_instruction=system_instruction,
            prompt=opening_prompt,
        )
