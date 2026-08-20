from pydantic import BaseModel

from app.schemas.session import SessionSummaryOut


class HomeOut(BaseModel):
    """FR-HOME-1..5."""
    current_metrics: dict[str, float]
    metrics_source: str  # "baseline" | "session" - which snapshot is being shown
    recent_sessions: list[SessionSummaryOut]


class AdminUserOut(BaseModel):
    id: str
    email: str
    onboarding_completed: bool
    sessions_count: int
    last_session_at: str | None


class AdminRetentionOut(BaseModel):
    """FR-ADM-2. Definition/window is Section 9 open item #8 - this shape
    covers the common cases (daily/weekly active, N-day retention) once
    product confirms which one they want."""
    window_days: int
    active_users: int
    total_users: int
    retention_rate: float
