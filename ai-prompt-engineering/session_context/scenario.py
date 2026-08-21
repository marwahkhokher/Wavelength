"""Structured parsing for raw user scenario descriptions."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field


class ParsedScenario(BaseModel):
    """Scenario facts extracted from user text without adding assumptions.

    This is intentionally distinct from the voice-infra ``ScenarioConfig``.
    ``ScenarioConfig`` is the existing live DTO; this model is the richer
    prompt-engineering parse that preserves information until that DTO grows.
    """

    raw_description: str
    conversation_type: str | None = None
    user_role: str | None = None
    counterpart_role: str | None = None
    goal: str | None = None
    context: str | None = None
    topics: list[str] = Field(default_factory=list)
    stakes: str | None = None
    constraints: list[str] = Field(default_factory=list)
    known_facts: list[str] = Field(default_factory=list)


def parse_scenario(raw_description: str) -> ParsedScenario:
    """Extract only narrow, explicitly supported scenario signals.

    The parser is deliberately conservative. It recognizes common scenario
    phrases needed by the current examples and leaves all other fields empty.
    """
    if not isinstance(raw_description, str) or not raw_description.strip():
        raise ValueError("raw_description must be a non-empty string")

    scenario = ParsedScenario(raw_description=raw_description)
    text = _normalized(raw_description)
    lower_text = text.casefold()

    if _contains_any(lower_text, "interview", "job interview"):
        scenario.conversation_type = "job_interview"
        scenario.user_role = "candidate"
        scenario.counterpart_role = "interviewer"
        scenario.context = "workplace conversation"
        scenario.known_facts.append("The scenario is an interview.")

        if "technical" in lower_text:
            scenario.topics.append("technical questions")
        if "behavioral" in lower_text:
            scenario.topics.append("behavioral questions")

    if _contains_any(lower_text, "ask my manager for a raise", "ask my manager for a salary increase"):
        scenario.conversation_type = "workplace_conversation"
        scenario.user_role = "employee"
        scenario.counterpart_role = "manager"
        scenario.goal = "Discuss a raise or salary increase with the manager."
        scenario.context = "workplace conversation"
        scenario.topics.append("salary discussion")
        scenario.known_facts.append("The user wants to ask their manager for a raise.")

    if _contains_any(lower_text, "talk to my friend", "talk with my friend"):
        scenario.conversation_type = "personal_conversation"
        scenario.counterpart_role = "friend"
        scenario.context = "personal conversation"
        if "cancell" in lower_text and "plan" in lower_text:
            scenario.goal = "Discuss the friend's repeated cancellations of plans."
            scenario.topics.append("cancelled plans")
            scenario.known_facts.append("The friend keeps cancelling plans.")

    return scenario


def _normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _contains_any(value: str, *phrases: str) -> bool:
    return any(phrase in value for phrase in phrases)