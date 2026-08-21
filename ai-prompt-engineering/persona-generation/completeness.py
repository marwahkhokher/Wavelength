"""
Completeness Checking — pure code, no LLM call.

Required/useful gap identification and scoring already happened inside
situation_extraction's single LLM call (see architecture doc Part 15 -
these were deliberately merged into one round trip). This module applies
the deterministic gate on top of that output: are all required gaps
resolved, and how many useful gaps fit this attempt's question budget?

Same SituationDraft in always produces the same ready/need_info decision -
no second LLM judgment call, so this is fully unit-testable with plain
fixtures and never inconsistent run to run for equivalent input.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from situation_extraction import SituationDraft

#: Max useful-tier questions asked in a single round (one completeness check).
USEFUL_QUESTIONS_PER_ROUND = 2

#: Max useful-tier questions asked across an entire generation attempt,
#: even over multiple question/answer rounds. Required-tier questions are
#: never subject to this cap - if information is required, it's asked
#: regardless of how many rounds have already happened.
MAX_USEFUL_QUESTIONS_TOTAL = 3


class CompletenessResult(BaseModel):
    status: Literal["ready", "need_info"]
    questions: list[str] = Field(default_factory=list)


def check_completeness(
    draft: SituationDraft, *, useful_questions_asked_so_far: int = 0
) -> CompletenessResult:
    required_gaps = [g for g in draft.gaps if g.tier == "required"]
    if required_gaps:
        return CompletenessResult(
            status="need_info", questions=[g.question for g in required_gaps]
        )

    remaining_budget = max(0, MAX_USEFUL_QUESTIONS_TOTAL - useful_questions_asked_so_far)
    if remaining_budget == 0:
        # Budget exhausted - proceed with whatever's known rather than ask forever.
        return CompletenessResult(status="ready")

    useful_gaps = sorted(
        (g for g in draft.gaps if g.tier == "useful"),
        key=lambda g: g.importance,
        reverse=True,
    )
    to_ask = useful_gaps[: min(USEFUL_QUESTIONS_PER_ROUND, remaining_budget)]

    if to_ask:
        return CompletenessResult(status="need_info", questions=[g.question for g in to_ask])

    return CompletenessResult(status="ready")
