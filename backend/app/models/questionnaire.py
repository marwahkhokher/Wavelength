import uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class QuestionnaireResponse(Base):
    """FR-ONB-1..8: one-time onboarding questionnaire, visible to Admin (FR-ONB-8)."""

    __tablename__ = "questionnaire_responses"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), unique=True, nullable=False
    )

    # Raw answers, keyed by question id. Shape: {"q1": {"type": "rating", "value": 4}, ...}
    # The exact 10-15 questions are Section 9 open item #2 - stored as JSON so
    # the question set can be finalized without a schema migration.
    answers: Mapped[dict] = mapped_column(JSON, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="questionnaire_response")
