from fastapi import APIRouter, Depends
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.database import get_db
from app.models.metrics import MetricSnapshot
from app.models.session import ConversationSession
from app.models.user import User
from app.schemas.home import HomeOut
from app.schemas.session import SessionSummaryOut

router = APIRouter(prefix="/home", tags=["home"])


@router.get("", response_model=HomeOut)
def get_home(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # FR-HOME-4/5: most recent snapshot - baseline until the first session
    # evaluation exists, then the latest session's values.
    # (Section 9 item #6: swap this for a running-average query if that's
    # what "current performance" ends up meaning - the data already supports it.)
    latest_snapshot = (
        db.query(MetricSnapshot)
        .filter(MetricSnapshot.user_id == user.id)
        .order_by(desc(MetricSnapshot.created_at))
        .first()
    )

    recent_sessions = (
        db.query(ConversationSession)
        .filter(ConversationSession.user_id == user.id)
        .order_by(desc(ConversationSession.created_at))
        .limit(10)
        .all()
    )

    return HomeOut(
        current_metrics=latest_snapshot.metrics if latest_snapshot else {},
        metrics_source=latest_snapshot.source.value if latest_snapshot else "none",
        recent_sessions=[SessionSummaryOut.model_validate(s) for s in recent_sessions],
    )
