"""Per-connection glue: STT events -> turn-taking -> AI service -> TTS.

One ``SessionOrchestrator`` is created per live websocket connection. It owns
no transport concerns (that's ``ws/router.py``) and no session bookkeeping
(that's ``SessionManager``) - it just wires the pipeline together and reacts
to barge-in by cancelling in-flight AI/TTS work.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from wavelength_voice.ai_service.client import AIServiceClient
from wavelength_voice.ai_service.contracts import (
    AITurnRequest,
    ConversationTurnDTO,
    PersonaConfig,
)
from wavelength_voice.session_state.manager import SessionManager
from wavelength_voice.voice_pipeline.stt import STTStream
from wavelength_voice.voice_pipeline.tts import TTSStream
from wavelength_voice.voice_pipeline.turn_taking import (
    InvalidTurnTransition,
    TurnState,
    TurnTakingController,
)

logger = logging.getLogger(__name__)

SendAudio = Callable[[bytes], Awaitable[None]]
SendEvent = Callable[[dict], Awaitable[None]]


class SessionOrchestrator:
    def __init__(
        self,
        session_id: str,
        persona: PersonaConfig,
        stt: STTStream,
        tts: TTSStream,
        ai_client: AIServiceClient,
        session_manager: SessionManager,
        send_audio: SendAudio,
        send_event: SendEvent,
    ) -> None:
        self._session_id = session_id
        self._persona = persona
        self._stt = stt
        self._tts = tts
        self._ai_client = ai_client
        self._session_manager = session_manager
        self._send_audio = send_audio
        self._send_event = send_event

        self.turns = TurnTakingController(on_barge_in=self._on_barge_in)
        self._stt_event_task: asyncio.Task[None] | None = None
        self._ai_task: asyncio.Task[None] | None = None
        self._tts_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        await self._stt.start()
        self._stt_event_task = asyncio.create_task(self._consume_stt_events())

    async def feed_audio(self, chunk: bytes) -> None:
        await self._stt.send_audio(chunk)

    async def stop(self) -> None:
        for task in (self._stt_event_task, self._ai_task, self._tts_task):
            if task is not None and not task.done():
                task.cancel()
        await self._stt.finish()

    def _on_barge_in(self, interrupted_state: TurnState, at) -> None:  # noqa: ANN001
        logger.info(
            "Barge-in on session %s (was %s)", self._session_id, interrupted_state
        )
        if self._ai_task is not None and not self._ai_task.done():
            self._ai_task.cancel()
        if self._tts_task is not None and not self._tts_task.done():
            self._tts_task.cancel()
        if interrupted_state == TurnState.AGENT_SPEAKING:
            asyncio.create_task(self._tts.cancel())

    async def _consume_stt_events(self) -> None:
        async for event in self._stt.events():
            try:
                if event.type == "speech_started":
                    self.turns.user_speech_started()
                elif event.type == "transcript" and event.is_final and event.text:
                    if self.turns.state == TurnState.WAITING_FOR_USER:
                        # Some STT providers can drop/coalesce the interim
                        # speech_started event; treat a final transcript as
                        # an implicit start-of-turn so we never lose it.
                        self.turns.user_speech_started()
                    self.turns.user_speech_ended(event.text)
                    self._ai_task = asyncio.create_task(
                        self._handle_user_utterance(event.text)
                    )
                elif event.type in ("error", "closed"):
                    return
            except InvalidTurnTransition:
                logger.warning(
                    "Ignoring out-of-sequence STT event %r for session %s in state %s",
                    event.type,
                    self._session_id,
                    self.turns.state,
                )

    async def _handle_user_utterance(self, transcript: str) -> None:
        session = self._session_manager.get_session(self._session_id)
        if session is None:
            return

        self._session_manager.append_turn(self._session_id, "user", transcript)
        history = [
            ConversationTurnDTO(role=t.role, text=t.text, turn_index=t.turn_index)
            for t in session.conversation_history
        ]
        request = AITurnRequest(
            session_id=self._session_id,
            user_id=session.user_id,
            persona=self._persona,
            transcript=transcript,
            conversation_history=history,
            turn_number=session.turn_number,
        )

        try:
            response = await self._ai_client.get_response(request)
        except asyncio.CancelledError:
            return  # barged into while the AI service was thinking

        if self.turns.state != TurnState.PROCESSING:
            return  # stale response for a turn the user has already moved past

        self.turns.agent_response_ready()
        self._session_manager.append_turn(self._session_id, "agent", response.reply_text)
        await self._send_event({"type": "agent_reply_text", "text": response.reply_text})

        self._tts_task = asyncio.create_task(self._stream_tts(response.reply_text))

        if response.end_session:
            await self._send_event({"type": "session_ended"})
            await self._session_manager.end_session(self._session_id)

    async def _stream_tts(self, text: str) -> None:
        try:
            async for chunk in self._tts.synthesize(text):
                await self._send_audio(chunk)
        except asyncio.CancelledError:
            return  # cancelled due to barge-in; controller state already moved on

        if self.turns.state == TurnState.AGENT_SPEAKING:
            self.turns.agent_speech_ended()
            await self._send_event({"type": "agent_speech_ended"})
