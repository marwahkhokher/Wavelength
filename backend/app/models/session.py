import uuid
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.enums import InputMethod, Mode, SessionStatus


class ConversationSession(Base):
    """
    One row per practice session, built up screen by screen
    (Mode -> Scenario -> Persona -> Live Session -> Evaluation),
    matching FR-HIST-2's required fields exactly.
    """

    __tablename__ = "conversation_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    status: Mapped[SessionStatus] = mapped_column(
        Enum(SessionStatus), default=SessionStatus.DRAFT, nullable=False
    )

    # --- Mode Selection (FR-MODE-*) ---
    mode: Mapped[Mode | None] = mapped_column(Enum(Mode), nullable=True)

    # --- Scenario Input (FR-SCEN-*) ---
    scenario_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    scenario_input_method: Mapped[InputMethod | None] = mapped_column(Enum(InputMethod), nullable=True)

    # --- Persona Input & Profile (FR-PERS-*) ---
    persona_description_raw: Mapped[str | None] = mapped_column(Text, nullable=True)
    persona_input_method: Mapped[InputMethod | None] = mapped_column(Enum(InputMethod), nullable=True)
    # AI-generated profile, e.g. {"name": ..., "role": ..., "personality_traits": [...],
    # "speech_style": ..., "background": ..., "goals": ...}. Exact fields are
    # Section 9 open item #3 - kept as JSON, edited in place by the user (FR-PERS-5),
    # then locked once persona_finalized flips to True (FR-PERS-6).
    persona_profile: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    persona_finalized: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # --- Session settings (FR-PERS-8/9) ---
    # Difficulty levels and duration options are Section 9 open items #4/#5.
    # difficulty stored as a string key (e.g. "easy"/"medium"/"hard") rather than
    # an enum so product-research can finalize the levels without a migration.
    difficulty: Mapped[str | None] = mapped_column(String, nullable=True)
    duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    extended_seconds: Mapped[int] = mapped_column(Integer, default=0, nullable=False)  # FR-SESS-3

    # --- Live session (FR-SESS-*) ---
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    audio_url: Mapped[str | None] = mapped_column(String, nullable=True)
    # Turn-by-turn transcript, captured for filler-word scoring (FR-SESS-8):
    # [{"speaker": "user"|"persona", "text": ..., "ts": ...}, ...]
    transcript: Mapped[list | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="sessions")
    evaluation = relationship(
        "Evaluation", back_populates="session", uselist=False, cascade="all, delete-orphan"
    )
    metric_snapshot = relationship("MetricSnapshot", back_populates="session", uselist=False)
