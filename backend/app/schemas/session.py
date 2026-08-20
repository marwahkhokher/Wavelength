import uuid
from datetime import datetime

from pydantic import BaseModel

from app.models.enums import InputMethod, Mode, SessionStatus


class CreateSessionRequest(BaseModel):
    """FR-MODE-1/3: starts a session with the chosen mode."""
    mode: Mode


class ScenarioInputRequest(BaseModel):
    """FR-SCEN-1..4."""
    scenario_text: str
    input_method: InputMethod


class PersonaInputRequest(BaseModel):
    """FR-PERS-1..3: triggers AI persona generation (FR-PERS-4)."""
    persona_description: str
    input_method: InputMethod


class PersonaProfileUpdateRequest(BaseModel):
    """FR-PERS-5: user edits the generated profile before finalizing."""
    persona_profile: dict


class SessionSettingsRequest(BaseModel):
    """FR-PERS-8/9."""
    difficulty: str
    duration_seconds: int


class SessionOut(BaseModel):
    id: uuid.UUID
    status: SessionStatus
    mode: Mode | None
    scenario_text: str | None
    persona_profile: dict | None
    persona_finalized: bool
    difficulty: str | None
    duration_seconds: int | None
    started_at: datetime | None
    ended_at: datetime | None
    created_at: datetime

    class Config:
        from_attributes = True


class SessionSummaryOut(BaseModel):
    """FR-HIST-1/2: lightweight shape for the previous-sessions list."""
    id: uuid.UUID
    mode: Mode | None
    scenario_text: str | None
    difficulty: str | None
    duration_seconds: int | None
    status: SessionStatus
    created_at: datetime

    class Config:
        from_attributes = True
