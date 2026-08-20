"""The single websocket endpoint: /ws/session/{session_id}.

Wire protocol (client <-> server):

* Client -> server: binary websocket frames are raw audio chunks (PCM16 at
  the configured sample rate); text frames are JSON control messages, e.g.
  ``{"type": "end_session"}``.
* Server -> client: binary frames are TTS audio chunks; text frames are JSON
  events - ``connected``, ``agent_reply_text``, ``agent_speech_ended``,
  ``session_taken_over``, ``session_ended``, ``error``.

This module owns transport (accept/receive/send/close) and delegates all
policy to ``SessionManager`` (reconnect/duplicate-tab) and
``SessionOrchestrator`` (STT -> turn-taking -> AI -> TTS).
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from wavelength_voice.ai_service.contracts import PersonaConfig
from wavelength_voice.app_state import AppState
from wavelength_voice.voice_pipeline.stt import WhisperSTTStream
from wavelength_voice.voice_pipeline.tts import ElevenLabsTTSStream
from wavelength_voice.ws.orchestrator import SessionOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter()

_DEFAULT_PERSONA = PersonaConfig(
    persona_id="default",
    name="Alex",
    scenario_prompt="A busy, mildly skeptical stakeholder in a project check-in.",
    difficulty="medium",
    traits=["direct", "impatient"],
)


@router.websocket("/ws/session/{session_id}")
async def session_websocket(
    websocket: WebSocket, session_id: str, user_id: str = Query(...)
) -> None:
    state: AppState = websocket.app.state.wavelength
    await websocket.accept()
    connection_id = str(uuid.uuid4())

    result = await state.session_manager.connect(session_id, connection_id, user_id)
    if not result.accepted:
        await websocket.send_json({"type": "error", "reason": result.reason})
        await websocket.close(code=4404)
        return

    if result.evicted_connection_id is not None:
        await _evict(state, result.evicted_connection_id)

    state.connections[connection_id] = websocket
    await websocket.send_json(
        {"type": "connected", "reason": result.reason, "session_id": session_id}
    )

    settings = state.settings
    stt = WhisperSTTStream(
        model_size=settings.whisper_model_size,
        language=settings.whisper_language,
        sample_rate=settings.whisper_sample_rate,
    )
    tts = ElevenLabsTTSStream(
        api_key=settings.elevenlabs_api_key,
        voice_id=settings.elevenlabs_voice_id,
        model_id=settings.elevenlabs_model_id,
        output_format=settings.elevenlabs_output_format,
    )

    orchestrator = SessionOrchestrator(
        session_id=session_id,
        persona=_DEFAULT_PERSONA,
        stt=stt,
        tts=tts,
        ai_client=state.ai_client,
        session_manager=state.session_manager,
        send_audio=websocket.send_bytes,
        send_event=websocket.send_json,
    )

    try:
        await orchestrator.start()
    except Exception:
        logger.exception("Failed to start voice pipeline for session %s", session_id)
        await websocket.send_json({"type": "error", "reason": "pipeline_start_failed"})
        await websocket.close(code=1011)
        state.connections.pop(connection_id, None)
        await state.session_manager.disconnect(session_id, connection_id)
        return

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if (audio := message.get("bytes")) is not None:
                await orchestrator.feed_audio(audio)
            elif (text := message.get("text")) is not None:
                await _handle_control_message(state, session_id, text)
    except WebSocketDisconnect:
        pass
    finally:
        await orchestrator.stop()
        state.connections.pop(connection_id, None)
        await state.session_manager.disconnect(session_id, connection_id)


async def _handle_control_message(state: AppState, session_id: str, text: str) -> None:
    import json

    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return
    if payload.get("type") == "end_session":
        await state.session_manager.end_session(session_id)


async def _evict(state: AppState, connection_id: str) -> None:
    evicted_ws = state.connections.pop(connection_id, None)
    if evicted_ws is None:
        return
    try:
        await evicted_ws.send_json({"type": "session_taken_over"})
        await evicted_ws.close(code=4409)
    except Exception:  # noqa: BLE001 - best-effort; the old tab may already be gone
        logger.debug("Could not cleanly close evicted connection %s", connection_id)
