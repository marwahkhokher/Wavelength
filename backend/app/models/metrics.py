import uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, Enum, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.enums import MetricSource


class MetricSnapshot(Base):
    """
    FR-MET-1..3: one consistent metric set, shown identically at onboarding,
    Home, evaluation, and coach mode.

    A snapshot is created:
      - once at signup, from the questionnaire (source=baseline)
      - once per completed session, from the evaluation (source=session)

    Home's "current performance" (FR-HOME-3/4/5) reads the most recent
    snapshot for the user. NOTE: whether "current" means latest snapshot,
    a running average, or a cumulative score is Section 9 open item #6 -
    this table supports all three since it keeps full history; the
    aggregation rule just needs to be decided at the query layer.
    """

    __tablename__ = "metric_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversation_sessions.id"), nullable=True
    )

    source: Mapped[MetricSource] = mapped_column(Enum(MetricSource), nullable=False)

    # {"clarity": 62, "empathy": 74, "filler_words": 8, ...}
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="metric_snapshots")
    session = relationship("ConversationSession", back_populates="metric_snapshot")
