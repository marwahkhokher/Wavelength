"""Pipeline Orchestrator (Taha's ownership).

Backward-compatible wrapper around ConversationOrchestrator for the sequential
turn pipeline connecting STT -> Tone -> Qwen Eval -> Prompt LLM -> ElevenLabs TTS.
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root.parent / "voice-tech-infra" / "src"))

from wavelength_voice.ai_service.contracts import (
    PromptLLMOutput,
    SessionContext,
    TurnRecord,
)

from .conversation_orchestrator import ConversationOrchestrator
from .interviewer_llm import InterviewerLLM
from .tts_stream import ElevenLabsTTSClient


class VoicePipelineOrchestrator:
    """Orchestrates the sequential turn pipeline across all AI team modules."""

    def __init__(self, session_context: SessionContext):
        self.session_context = session_context
        self._orchestrator = ConversationOrchestrator(
            session_context=session_context,
            interviewer_llm=InterviewerLLM(),
            tts_client=ElevenLabsTTSClient(),
            use_mock_providers=False,
        )

    @property
    def conversation_history(self) -> list[TurnRecord]:
        return self._orchestrator.turn_records

    @property
    def turn_counter(self) -> int:
        return self._orchestrator.turn_counter

    async def process_user_turn(self, audio_bytes: bytes) -> PromptLLMOutput:
        """Executes the full sequential pipeline for one user turn:
        Audio -> STT + Tone -> Qwen Deep Evaluation -> Prompt LLM -> ElevenLabs TTS.
        """
        output, _ = await self._orchestrator.process_turn_audio(audio_bytes)
        return output
