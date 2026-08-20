import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.database import get_db
from app.models.enums import MetricSource, SessionStatus
from app.models.evaluation import Evaluation
from app.models.metrics import MetricSnapshot
from app.models.session import ConversationSession
from app.models.user import User
from app.schemas.evaluation import EvaluationOut
from app.schemas.session import (
    CreateSessionRequest,
    PersonaInputRequest,
    PersonaProfileUpdateRequest,
    ScenarioInputRequest,
    SessionOut,
    SessionSettingsRequest,
    SessionSummaryOut,
)
from app.services.evaluation import generate_coach_recommendations, score_transcript
from app.services.persona_generation import generate_persona_profile

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _get_owned_session(session_id: uuid.UUID, user: User, db: Session) -> ConversationSession:
    session = db.get(ConversationSession, session_id)
    if not session or session.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    return session


@router.post("", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
def create_session(
    payload: CreateSessionRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """FR-MODE-1..4: Start a Talk -> Mode Selection."""
    session = ConversationSession(user_id=user.id, mode=payload.mode, status=SessionStatus.DRAFT)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.patch("/{session_id}/scenario", response_model=SessionOut)
def set_scenario(
    session_id: uuid.UUID,
    payload: ScenarioInputRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """FR-SCEN-1..5."""
    session = _get_owned_session(session_id, user, db)
    session.scenario_text = payload.scenario_text
    session.scenario_input_method = payload.input_method
    db.commit()
    db.refresh(session)
    return session


@router.post("/{session_id}/persona/generate", response_model=SessionOut)
def generate_persona(
    session_id: uuid.UUID,
    payload: PersonaInputRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """FR-PERS-1..4: user's description in, AI-generated profile out."""
    session = _get_owned_session(session_id, user, db)
    if not session.scenario_text:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Scenario must be set before persona")

    session.persona_description_raw = payload.persona_description
    session.persona_input_method = payload.input_method
    session.persona_profile = generate_persona_profile(
        session.scenario_text, payload.persona_description, session.mode
    )
    session.persona_finalized = False
    db.commit()
    db.refresh(session)
    return session


@router.patch("/{session_id}/persona", response_model=SessionOut)
def update_persona(
    session_id: uuid.UUID,
    payload: PersonaProfileUpdateRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """FR-PERS-5: user edits the generated profile."""
    session = _get_owned_session(session_id, user, db)
    if session.persona_finalized:
        raise HTTPException(status.HTTP_409_CONFLICT, "Persona already finalized")
    session.persona_profile = payload.persona_profile
    db.commit()
    db.refresh(session)
    return session


@router.post("/{session_id}/persona/finalize", response_model=SessionOut)
def finalize_persona(
    session_id: uuid.UUID, user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """FR-PERS-6/7."""
    session = _get_owned_session(session_id, user, db)
    if not session.persona_profile:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No persona profile to finalize")
    session.persona_finalized = True
    db.commit()
    db.refresh(session)
    return session


@router.patch("/{session_id}/settings", response_model=SessionOut)
def set_session_settings(
    session_id: uuid.UUID,
    payload: SessionSettingsRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """FR-PERS-8/9/10."""
    session = _get_owned_session(session_id, user, db)
    session.difficulty = payload.difficulty
    session.duration_seconds = payload.duration_seconds
    db.commit()
    db.refresh(session)
    return session


@router.post("/{session_id}/begin", response_model=SessionOut)
def begin_session(
    session_id: uuid.UUID, user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """
    FR-PERS-11: kicks off the live session. Actual audio streaming happens
    over the WebSocket in app/ws/live_session.py - this just flips state so
    the client knows it's clear to open the socket.
    """
    session = _get_owned_session(session_id, user, db)
    if not session.persona_finalized:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Persona must be finalized first")
    if not session.duration_seconds or not session.difficulty:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Difficulty and duration must be set first")

    session.status = SessionStatus.IN_PROGRESS
    session.started_at = datetime.utcnow()
    db.commit()
    db.refresh(session)
    return session


@router.post("/{session_id}/end", response_model=EvaluationOut)
def end_session(
    session_id: uuid.UUID, user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """
    FR-SESS-7, FR-EVAL-1..5, FR-COACH-1..3: fires whether the timer expired
    or the user ended it early (both paths call this same endpoint), scores
    the transcript, and returns the evaluation + coach recommendations.
    """
    session = _get_owned_session(session_id, user, db)
    if session.status != SessionStatus.IN_PROGRESS:
        raise HTTPException(status.HTTP_409_CONFLICT, "Session is not in progress")

    session.status = SessionStatus.ENDED
    session.ended_at = datetime.utcnow()

    transcript = session.transcript or []
    metrics = score_transcript(transcript)
    recommendations = generate_coach_recommendations(metrics)

    evaluation = Evaluation(session_id=session.id, metrics=metrics, coach_recommendations=recommendations)
    db.add(evaluation)
    db.add(MetricSnapshot(
        user_id=user.id, session_id=session.id, source=MetricSource.SESSION, metrics=metrics
    ))
    session.status = SessionStatus.EVALUATED
    db.commit()
    db.refresh(evaluation)
    return evaluation


@router.get("/{session_id}/evaluation", response_model=EvaluationOut)
def get_evaluation(
    session_id: uuid.UUID, user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    session = _get_owned_session(session_id, user, db)
    if not session.evaluation:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Evaluation not ready")
    return session.evaluation


@router.get("", response_model=list[SessionSummaryOut])
def list_sessions(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """FR-HIST-1/2."""
    return (
        db.query(ConversationSession)
        .filter(ConversationSession.user_id == user.id)
        .order_by(desc(ConversationSession.created_at))
        .all()
    )


@router.get("/{session_id}", response_model=SessionOut)
def get_session(
    session_id: uuid.UUID, user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    return _get_owned_session(session_id, user, db)
