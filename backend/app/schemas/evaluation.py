import uuid
from datetime import datetime

from pydantic import BaseModel


class CoachRecommendation(BaseModel):
    metric: str
    observation: str
    recommendation: str


class EvaluationOut(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    metrics: dict[str, float]
    coach_recommendations: list[CoachRecommendation]
    created_at: datetime

    class Config:
        from_attributes = True
