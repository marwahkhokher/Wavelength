from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.database import get_db
from app.models.enums import MetricSource
from app.models.metrics import MetricSnapshot
from app.models.questionnaire import QuestionnaireResponse
from app.models.user import User
from app.schemas.onboarding import QuestionnaireSubmitRequest, QuestionnaireSubmitResponse, QuestionsOut
from app.services.questions import AGREEMENT_TO_SCORE, QUESTIONS

router = APIRouter(prefix="/onboarding", tags=["onboarding"])


@router.get("/questions", response_model=QuestionsOut)
def get_questions():
    """FR-ONB-1..4."""
    return QuestionsOut(questions=QUESTIONS)


@router.post("/responses", response_model=QuestionnaireSubmitResponse)
def submit_responses(
    payload: QuestionnaireSubmitRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if user.onboarding_completed:
        raise HTTPException(status.HTTP_409_CONFLICT, "Onboarding already completed")

    questions_by_id = {q.id: q for q in QUESTIONS}
    per_metric_scores: dict[str, list[float]] = defaultdict(list)

    for question_id, raw_answer in payload.answers.items():
        question = questions_by_id.get(question_id)
        if not question:
            continue
        score = (
            AGREEMENT_TO_SCORE[raw_answer]
            if question.type == "agreement_scale"
            else (int(raw_answer) - 1) / 4 * 100  # 1-5 rating -> 0-100
        )
        per_metric_scores[question.maps_to_metric].append(score)

    baseline_metrics = {
        metric: round(sum(scores) / len(scores), 1) for metric, scores in per_metric_scores.items()
    }
    # filler_words can't be measured from a questionnaire - neutral baseline
    # until the user's first session produces a real, speech-derived value.
    baseline_metrics.setdefault("filler_words", 0)

    db.add(QuestionnaireResponse(user_id=user.id, answers=payload.answers))
    db.add(MetricSnapshot(user_id=user.id, source=MetricSource.BASELINE, metrics=baseline_metrics))
    user.onboarding_completed = True  # FR-AUTH-4 / FR-ONB-7
    db.commit()

    return QuestionnaireSubmitResponse(baseline_metrics=baseline_metrics)
