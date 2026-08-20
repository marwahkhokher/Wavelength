from typing import Literal

from pydantic import BaseModel


class QuestionOption(BaseModel):
    id: str
    label: str


class Question(BaseModel):
    """
    FR-ONB-3/4: supports rating questions and agreement-scale questions.
    The concrete 10-15 questions and their metric mapping are Section 9
    open item #2 - `questions.py` in services/ is the single place that
    needs updating once product-research finalizes them.
    """

    id: str
    text: str
    type: Literal["rating", "agreement_scale"]
    options: list[QuestionOption]
    maps_to_metric: str  # which metric this question contributes to


class QuestionsOut(BaseModel):
    questions: list[Question]


class QuestionnaireSubmitRequest(BaseModel):
    # {"q1": "4", "q2": "strongly_agree", ...}
    answers: dict[str, str]


class QuestionnaireSubmitResponse(BaseModel):
    baseline_metrics: dict[str, float]
