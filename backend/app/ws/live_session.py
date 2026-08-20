"""
FR-SESS-1..8: the live conversation.

This is deliberately an interface, not a finished voice pipeline. FR-SESS-4
("cut the AI off mid-speech") means whoever ends up wired in here has to
support real-time interruption - that's a voice-tech-infra decision
(OpenAI Realtime API vs ElevenLabs Conversational AI vs a custom
STT->LLM->TTS pipeline with your own barge-in detection), not something to
lock in from the backend side. Confirm that before building past this stub.

Client -> server control messages (JSON over the socket):
  {"type": "mute", "value": true|false}       FR-SESS-1
  {"type": "interrupt"}                        FR-SESS-4 - cut the AI off
  {"type": "extend", "seconds": 60}            FR-SESS-3
  {"type": "end"}                              FR-SESS-2
  {"type": "audio_chunk", "data": "<base64>"}  streamed user audio

Server -> client messages:
  {"type": "persona_audio_chunk", "data": "<base64>"}
  {"type": "transcript_turn", "speaker": "user"|"persona", "text": "..."}
  {"type": "session_ended", "reason": "timer"|"user_ended"}
"""

import json
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models.enums import SessionStatus
from app.models.session import ConversationSession

router = APIRouter()


class VoiceProvider:
    """Swap this out for the confirmed realtime voice provider integration."""

    async def start(self, persona_profile: dict, tone_system_prompt: str, difficulty: str):
        raise NotImplementedError

    async def push_user_audio(self, chunk: bytes):
        raise NotImplementedError

    async def interrupt(self):
        """FR-SESS-4."""
        raise NotImplementedError

    async def stop(self):
        raise NotImplementedError


@router.websocket("/ws/sessions/{session_id}")
async def live_session_socket(websocket: WebSocket, session_id: uuid.UUID):
    await websocket.accept()
    db: Session = SessionLocal()

    try:
        session = db.get(ConversationSession, session_id)
        if not session or session.status != SessionStatus.IN_PROGRESS:
            await websocket.close(code=4404, reason="Session not found or not in progress")
            return

        # provider = get_voice_provider(settings.voice_provider)
        # await provider.start(session.persona_profile, build_tone_system_prompt(session.mode), session.difficulty)

        transcript: list[dict] = session.transcript or []

        while True:
            raw = await websocket.receive_text()
            message = json.loads(raw)
            msg_type = message.get("type")

            if msg_type == "mute":
                pass  # FR-SESS-1: client-side stops sending audio_chunk while muted

            elif msg_type == "interrupt":
                # FR-SESS-4: await provider.interrupt()
                pass

            elif msg_type == "extend":
                session.extended_seconds += int(message.get("seconds", 0))  # FR-SESS-3
                db.commit()

            elif msg_type == "audio_chunk":
                # FR-SESS-8: forward to provider for STT, append recognized
                # text to `transcript` as it comes back.
                pass

            elif msg_type == "end":
                break

        session.transcript = transcript
        db.commit()
        await websocket.send_json({"type": "session_ended", "reason": "user_ended"})

    except WebSocketDisconnect:
        pass
    finally:
        db.close()
