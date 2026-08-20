from datetime import datetime, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from app.core.deps import require_admin
from app.database import get_db
from app.models.session import ConversationSession
from app.models.user import User
from app.schemas.home import AdminRetentionOut, AdminUserOut

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


@router.get("/users", response_model=list[AdminUserOut])
def list_users(db: Session = Depends(get_db)):
    """FR-ADM-1."""
    users = db.query(User).all()
    out = []
    for u in users:
        last_session = (
            db.query(ConversationSession)
            .filter(ConversationSession.user_id == u.id)
            .order_by(desc(ConversationSession.created_at))
            .first()
        )
        sessions_count = (
            db.query(func.count(ConversationSession.id))
            .filter(ConversationSession.user_id == u.id)
            .scalar()
        )
        out.append(AdminUserOut(
            id=str(u.id),
            email=u.email,
            onboarding_completed=u.onboarding_completed,
            sessions_count=sessions_count,
            last_session_at=last_session.created_at.isoformat() if last_session else None,
        ))
    return out


@router.get("/retention", response_model=AdminRetentionOut)
def retention(window_days: int = 7, db: Session = Depends(get_db)):
    """
    FR-ADM-2. Section 9 open item #8: "retention" isn't defined yet (DAU/WAU?
    N-day return rate?). This implements the simplest reasonable definition -
    fraction of users with a session in the last `window_days` - swap the
    query once product confirms what they actually want measured.
    """
    total_users = db.query(func.count(User.id)).scalar() or 0
    cutoff = datetime.utcnow() - timedelta(days=window_days)
    active_users = (
        db.query(func.count(func.distinct(ConversationSession.user_id)))
        .filter(ConversationSession.created_at >= cutoff)
        .scalar()
        or 0
    )
    rate = (active_users / total_users) if total_users else 0.0
    return AdminRetentionOut(
        window_days=window_days, active_users=active_users, total_users=total_users, retention_rate=round(rate, 3)
    )
