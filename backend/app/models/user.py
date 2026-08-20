import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.enums import Gender


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)

    gender: Mapped[Gender | None] = mapped_column(Enum(Gender), nullable=True)  # FR-ONB-5

    # FR-AUTH-4: per-user flag so onboarding is shown exactly once
    onboarding_completed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # FR-ADM-3: admin accounts are distinct from end users
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    questionnaire_response = relationship(
        "QuestionnaireResponse", back_populates="user", uselist=False, cascade="all, delete-orphan"
    )
    sessions = relationship("ConversationSession", back_populates="user", cascade="all, delete-orphan")
    metric_snapshots = relationship("MetricSnapshot", back_populates="user", cascade="all, delete-orphan")
