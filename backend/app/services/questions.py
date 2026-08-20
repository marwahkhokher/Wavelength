"""
FR-ONB-1..5: 10-15 questions covering personality, emotional intelligence,
emotional awareness, humour level, plus gender.

Section 9 open item #2: the exact questions and their mapping to metrics are
NOT yet confirmed by product-research. This is a placeholder set, structured
so the mapping is explicit and swapping in the final list is a data change,
not a code change.

Rating questions: options are "1".."5" (Likert).
Agreement scale: options are the 5-point agree/disagree scale (FR-ONB-4).
"""

from app.schemas.onboarding import Question, QuestionOption

RATING_OPTIONS = [QuestionOption(id=str(i), label=str(i)) for i in range(1, 6)]

AGREEMENT_OPTIONS = [
    QuestionOption(id="strongly_disagree", label="Strongly Disagree"),
    QuestionOption(id="disagree", label="Disagree"),
    QuestionOption(id="neutral", label="Neutral"),
    QuestionOption(id="agree", label="Agree"),
    QuestionOption(id="strongly_agree", label="Strongly Agree"),
]

AGREEMENT_TO_SCORE = {
    "strongly_disagree": 0,
    "disagree": 25,
    "neutral": 50,
    "agree": 75,
    "strongly_agree": 100,
}

QUESTIONS: list[Question] = [
    Question(
        id="q1", text="I find it easy to say exactly what I mean.",
        type="agreement_scale", options=AGREEMENT_OPTIONS, maps_to_metric="clarity",
    ),
    Question(
        id="q2", text="People rarely ask me to repeat or clarify what I said.",
        type="agreement_scale", options=AGREEMENT_OPTIONS, maps_to_metric="clarity",
    ),
    Question(
        id="q3", text="I can usually tell how someone is feeling before they say it.",
        type="agreement_scale", options=AGREEMENT_OPTIONS, maps_to_metric="empathy",
    ),
    Question(
        id="q4", text="Rate how often you adjust your tone based on the other person's mood.",
        type="rating", options=RATING_OPTIONS, maps_to_metric="empathy",
    ),
    Question(
        id="q5", text="I notice my own emotions as they happen, not after the fact.",
        type="agreement_scale", options=AGREEMENT_OPTIONS, maps_to_metric="emotional_awareness",
    ),
    Question(
        id="q6", text="Rate your comfort level naming what you're feeling out loud.",
        type="rating", options=RATING_OPTIONS, maps_to_metric="emotional_awareness",
    ),
    Question(
        id="q7", text="I use humour comfortably in conversations with strangers.",
        type="agreement_scale", options=AGREEMENT_OPTIONS, maps_to_metric="humour",
    ),
    Question(
        id="q8", text="Rate how often your jokes land the way you intended.",
        type="rating", options=RATING_OPTIONS, maps_to_metric="humour",
    ),
]
