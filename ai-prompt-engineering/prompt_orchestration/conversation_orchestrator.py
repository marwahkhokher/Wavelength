"""Central Conversation Orchestrator (Taha's ownership).

Coordinates the multi-module AI workflow:
  1. Input Handling (Audio or Text)
  2. Transcript Provider (Person 2 - Areej)
  3. Tone & Emotion Provider (Person 3 - Zaid)
  4. Qwen Deep Evaluation Provider (Person 4 - Ahmed)
  5. Session State & Structured Memory Management
  6. Persona Consistency & Modular Prompt Assembly
  7. Interviewer LLM Response Generation (Person 5 - Taha)
  8. Streaming ElevenLabs TTS with Instant Barge-In Cancellation

Supports pluggable Provider interfaces so mock and real models can be swapped
instantly without modifying the orchestration logic.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from collections.abc import AsyncGenerator
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root.parent / "voice-tech-infra" / "src"))

from wavelength_voice.ai_service.contracts import (
    AITurnRequest,
    AITurnResponse,
    PerTurnEvaluation,
    PromptLLMInput,
    PromptLLMOutput,
    SessionContext,
    STTResult,
    ToneResult,
    TurnRecord,
)

from .config import get_ai_settings
from .conversation_memory import ConversationMemory
from .interviewer_llm import InterviewerLLM
from .models import ConversationState, SessionState
from .providers.base import (
    AnalysisProvider,
    SessionContextProvider,
    ToneProvider,
    TranscriptProvider,
)
from .providers.mock_providers import (
    MockAnalysisProvider,
    MockSessionContextProvider,
    MockToneProvider,
    MockTranscriptProvider,
)
from .providers.real_providers import (
    QwenAnalysisProvider,
    RealSessionContextProvider,
    RealToneProvider,
    WhisperTranscriptProvider,
)
from .session_manager import SessionManager
from .tts_stream import BaseTTSClient, ElevenLabsTTSClient

logger = logging.getLogger(__name__)


class ConversationOrchestrator:
    """Central orchestrator managing full session lifecycle, AI turns, and real-time streaming."""

    def __init__(
        self,
        session_context: SessionContext | None = None,
        transcript_provider: TranscriptProvider | None = None,
        tone_provider: ToneProvider | None = None,
        analysis_provider: AnalysisProvider | None = None,
        session_context_provider: SessionContextProvider | None = None,
        interviewer_llm: InterviewerLLM | None = None,
        tts_client: BaseTTSClient | None = None,
        session_manager: SessionManager | None = None,
        use_mock_providers: bool | None = None,
    ) -> None:
        settings = get_ai_settings()
        is_mock = use_mock_providers if use_mock_providers is not None else (settings.provider_mode == "mock")

        # Session context
        self.session_context = session_context
        self.session_context_provider = session_context_provider or (
            MockSessionContextProvider() if is_mock else RealSessionContextProvider()
        )

        # Teammate providers
        self.transcript_provider = transcript_provider or (
            MockTranscriptProvider() if is_mock else WhisperTranscriptProvider()
        )
        self.tone_provider = tone_provider or (
            MockToneProvider() if is_mock else RealToneProvider()
        )
        self.analysis_provider = analysis_provider or (
            MockAnalysisProvider() if is_mock else QwenAnalysisProvider()
        )

        # Memory & LLM
        self.memory_manager = ConversationMemory(
            max_recent_turns=settings.max_recent_turns,
            max_turns_before_summary=settings.max_history_turns_before_summary,
        )
        self.prompt_llm = interviewer_llm or InterviewerLLM(memory_manager=self.memory_manager)
        self.tts_client = tts_client or ElevenLabsTTSClient()
        self.session_manager = session_manager or SessionManager()

        # Turn history
        self.turn_records: list[TurnRecord] = []
        self.turn_counter = 0
        self._current_tts_task: asyncio.Task | None = None

    def get_or_create_state(self, session_id: str, user_id: str) -> ConversationState:
        """Retrieves or creates in-memory conversation state for this session."""
        return self.session_manager.get_or_create_session(session_id, user_id)

    async def initialize_session(
        self,
        session_id: str,
        user_id: str,
        mode: str = "professional",
        scenario_title: str = "Interview Practice",
        scenario_description: str = "Live interview roleplay",
        persona_name: str = "Alex",
        persona_role: str = "Hiring Manager",
        difficulty: str = "medium",
        duration_seconds: int = 300,
    ) -> tuple[SessionContext, str]:
        """Initializes a new session and generates the persona's opening turn."""
        self.session_context = self.session_context_provider.build_context(
            session_id=session_id,
            user_id=user_id,
            mode=mode,
            scenario_title=scenario_title,
            scenario_description=scenario_description,
            persona_name=persona_name,
            persona_role=persona_role,
            difficulty=difficulty,
            duration_seconds=duration_seconds,
        )

        state = self.get_or_create_state(session_id, user_id)
        state.state = SessionState.ACTIVE

        opening_text = await self.prompt_llm.generate_opening(
            session_context=self.session_context,
            conversation_state=state,
        )
        state.add_message("assistant", opening_text)

        logger.info("Session %s initialized. Opening: %s", session_id, opening_text)
        return self.session_context, opening_text

    async def process_turn_audio(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000,
    ) -> tuple[PromptLLMOutput, PerTurnEvaluation]:
        """Executes full turn pipeline from raw user audio bytes:
        Audio -> STT -> Tone Analysis -> Qwen Deep Evaluation -> Interviewer LLM.
        """
        t0 = time.perf_counter()
        self.turn_counter += 1

        if self.session_context is None:
            self.session_context = self.session_context_provider.build_context(
                session_id="default_session",
                user_id="default_user",
                mode="professional",
                scenario_title="Roleplay Practice",
                scenario_description="General communication practice",
                persona_name="Alex",
                persona_role="Manager",
            )

        state = self.get_or_create_state(
            self.session_context.session_id, self.session_context.user_id
        )

        # Step 1: STT & Filler Detection (Person 2 - Areej)
        stt_result: STTResult = await self.transcript_provider.transcribe(
            audio_bytes, sample_rate
        )
        state.add_message("user", stt_result.transcript)

        # Step 2: Parallel execution of Tone Analysis & Qwen Deep Evaluation
        tone_task = self.tone_provider.analyze_tone(
            audio_bytes=audio_bytes,
            word_count=stt_result.total_words,
            duration_sec=stt_result.utterance_duration_sec,
        )
        tone_result: ToneResult = await tone_task

        eval_task = self.analysis_provider.evaluate_turn(
            turn_index=self.turn_counter,
            stt_result=stt_result,
            tone_result=tone_result,
        )
        qwen_eval: PerTurnEvaluation = await eval_task

        # Step 3: Interviewer LLM Response Generation (Person 5 - Taha)
        llm_input = PromptLLMInput(
            session_context=self.session_context,
            current_stt=stt_result,
            current_tone=tone_result,
            conversation_history=self.turn_records,
        )
        llm_output = await self.prompt_llm.generate_next_turn(
            payload=llm_input,
            qwen_eval=qwen_eval,
            conversation_state=state,
        )
        state.add_message("assistant", llm_output.reply_text)

        # Step 4: Record history
        record = TurnRecord(
            turn_index=self.turn_counter,
            user_transcript=stt_result.transcript,
            user_tone=tone_result.primary_emotion,
            ai_response=llm_output.reply_text,
            eval_scores=qwen_eval.scores,
        )
        self.turn_records.append(record)

        total_latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Turn %d processed in %.1fms (STT=%.1fms, User='%s')",
            self.turn_counter,
            total_latency_ms,
            stt_result.stt_latency_ms,
            stt_result.transcript[:30],
        )

        return llm_output, qwen_eval

    async def handle_ai_turn_request(self, request: AITurnRequest) -> AITurnResponse:
        """Handles wire format AITurnRequest from voice-tech-infra's HTTPAIServiceClient."""
        t0 = time.perf_counter()

        # Build SessionContext from request payload
        session_ctx = SessionContext(
            session_id=request.session_id,
            user_id=request.user_id,
            mode="professional" if "professional" in request.persona.scenario_prompt.lower() else "personal",
            difficulty=request.persona.difficulty,
            scenario={
                "title": f"Session with {request.persona.name}",
                "description": request.persona.scenario_prompt,
                "setting": "Live Voice Dialogue",
            },
            persona={
                "name": request.persona.name,
                "role_description": request.persona.scenario_prompt,
                "communication_style": "Professional workplace norms" if request.persona.difficulty != "easy" else "Casual, cooperative",
                "attitude": "firm" if request.persona.difficulty == "hard" else "receptive",
                "tone_traits": request.persona.traits,
            },
        )

        state = self.get_or_create_state(request.session_id, request.user_id)
        state.add_message("user", request.transcript)

        # Build synthetic STT & Tone from text
        words = request.transcript.split()
        stt_result = STTResult(
            transcript=request.transcript,
            is_final=True,
            total_words=len(words),
            filler_word_count=0,
            utterance_duration_sec=max(1.0, len(words) * 0.4),
        )

        # Quick simulated background eval
        dummy_audio = b"\x00" * 3200
        tone_result = await self.tone_provider.analyze_tone(
            dummy_audio, word_count=len(words), duration_sec=stt_result.utterance_duration_sec
        )
        qwen_eval = await self.analysis_provider.evaluate_turn(
            turn_index=request.turn_number,
            stt_result=stt_result,
            tone_result=tone_result,
        )

        llm_input = PromptLLMInput(
            session_context=session_ctx,
            current_stt=stt_result,
            current_tone=tone_result,
            conversation_history=self.turn_records,
        )

        llm_output = await self.prompt_llm.generate_next_turn(
            payload=llm_input,
            qwen_eval=qwen_eval,
            conversation_state=state,
        )
        state.add_message("assistant", llm_output.reply_text)

        latency_ms = int((time.perf_counter() - t0) * 1000)

        return AITurnResponse(
            reply_text=llm_output.reply_text,
            persona_state={
                "disposition": llm_output.persona_state_update or "active",
                "difficulty": session_ctx.difficulty,
            },
            end_session=llm_output.end_session,
            feedback_hint=qwen_eval.coach_tip,
            latency_ms=latency_ms,
        )

    def trigger_barge_in(self) -> None:
        """Called when user starts speaking mid-AI utterance ('Cut AI off')."""
        logger.info("Barge-in triggered: canceling current TTS audio generation.")
        self.tts_client.trigger_interruption()
        if self._current_tts_task and not self._current_tts_task.done():
            self._current_tts_task.cancel()

    async def stream_audio_response(self, text: str) -> AsyncGenerator[bytes, None]:
        """Streams audio chunks to client with instant interruption support."""
        async for chunk in self.tts_client.stream_audio_response(text):
            yield chunk
