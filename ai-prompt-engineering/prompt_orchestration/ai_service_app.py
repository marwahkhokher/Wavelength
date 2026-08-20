"""AI Service FastAPI Application (Taha's ownership).

Exposes the HTTP API consumed by the voice-tech-infra WebSocket server and
frontend applications.

Endpoints:
  - POST /v1/turn: Process finalized STT transcript and return AITurnResponse (wire contract)
  - POST /v1/session/start: Initialize a new scenario session and return opening turn
  - GET  /v1/session/{id}: Get current conversation state & structured memory
  - POST /v1/session/{id}/end: Complete a session and finalize history
  - POST /v1/session/{id}/interrupt: Signal a barge-in event
  - GET  /health: Service health check
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root.parent / "voice-tech-infra" / "src"))

from wavelength_voice.ai_service.contracts import AITurnRequest, AITurnResponse

from .config import get_ai_settings
from .conversation_orchestrator import ConversationOrchestrator
from .session_manager import SessionManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_service")

app = FastAPI(
    title="Wavelength AI Service",
    description="Roleplay Persona LLM, Prompt Generation, and Real-Time Orchestration Service",
    version="1.0.0",
)

# Global session manager and orchestrator instance
session_manager = SessionManager()
orchestrator = ConversationOrchestrator(session_manager=session_manager)


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------

class StartSessionRequest(BaseModel):
    session_id: str
    user_id: str
    mode: str = "professional"
    scenario_title: str = "Job Interview Practice"
    scenario_description: str = "Live scenario dialogue"
    persona_name: str = "Alex"
    persona_role: str = "Senior Engineering Manager"
    difficulty: str = "medium"
    duration_seconds: int = 300


class StartSessionResponse(BaseModel):
    session_id: str
    opening_reply: str
    persona_name: str
    difficulty: str


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "wavelength-ai-prompt-engineering"}


@app.post("/v1/turn", response_model=AITurnResponse)
async def process_turn(request: AITurnRequest) -> AITurnResponse:
    """Primary wire contract endpoint consumed by voice-tech-infra HTTPAIServiceClient."""
    try:
        response = await orchestrator.handle_ai_turn_request(request)
        return response
    except Exception as e:
        logger.exception("Error processing turn for session %s: %s", request.session_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Turn processing error: {e}",
        ) from e


@app.post("/v1/session/start", response_model=StartSessionResponse)
async def start_session(payload: StartSessionRequest) -> StartSessionResponse:
    """Initializes a new session and generates the persona's opening turn."""
    try:
        ctx, opening = await orchestrator.initialize_session(
            session_id=payload.session_id,
            user_id=payload.user_id,
            mode=payload.mode,
            scenario_title=payload.scenario_title,
            scenario_description=payload.scenario_description,
            persona_name=payload.persona_name,
            persona_role=payload.persona_role,
            difficulty=payload.difficulty,
            duration_seconds=payload.duration_seconds,
        )
        return StartSessionResponse(
            session_id=payload.session_id,
            opening_reply=opening,
            persona_name=ctx.persona.name,
            difficulty=ctx.difficulty,
        )
    except Exception as e:
        logger.exception("Error starting session %s: %s", payload.session_id, e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e


@app.get("/v1/session/{session_id}")
async def get_session(session_id: str) -> dict:
    """Returns conversation state, structured memory, and turn count for a session."""
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "state": session.state,
        "current_turn": session.current_turn,
        "persona_disposition": session.persona_state.model_dump(),
        "structured_memory": session.structured_memory.model_dump(),
        "history_count": len(session.conversation_history),
    }


@app.post("/v1/session/{session_id}/end")
async def end_session(session_id: str) -> dict:
    """Marks a session as completed."""
    session = session_manager.end_session(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return {"status": "completed", "session_id": session_id, "total_turns": session.current_turn}


@app.post("/v1/session/{session_id}/interrupt")
async def interrupt_session(session_id: str) -> dict:
    """Handles barge-in event by interrupting any active TTS generation."""
    orchestrator.trigger_barge_in()
    return {"status": "interrupted", "session_id": session_id}


def run_server() -> None:
    """Entrypoint to run the AI Service with uvicorn."""
    import uvicorn
    settings = get_ai_settings()
    uvicorn.run(app, host=settings.ai_service_host, port=settings.ai_service_port)


if __name__ == "__main__":
    run_server()
