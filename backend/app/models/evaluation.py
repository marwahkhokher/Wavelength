import uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Evaluation(Base):
    """FR-EVAL-*, FR-COACH-*: one per session, scored on the standard metric set."""

    __tablename__ = "evaluations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversation_sessions.id"), unique=True, nullable=False
    )

    # Same shape/keys as MetricSnapshot.metrics - FR-MET-1 (identical metric set everywhere)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False)

    # FR-COACH-2/3: actionable recommendations derived from this session's metrics
    # e.g. [{"metric": "filler_words", "observation": ..., "recommendation": ...}, ...]
    coach_recommendations: Mapped[list] = mapped_column(JSON, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    session = relationship("ConversationSession", back_populates="evaluation")
