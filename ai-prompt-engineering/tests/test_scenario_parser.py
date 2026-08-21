from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-tech-infra" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from session_context.scenario import parse_scenario


def test_final_round_internship_interview() -> None:
    raw = "Final-round software engineering internship interview with technical and behavioral questions."

    scenario = parse_scenario(raw)

    assert scenario.raw_description == raw
    assert scenario.conversation_type == "job_interview"
    assert scenario.user_role == "candidate"
    assert scenario.counterpart_role == "interviewer"
    assert scenario.topics == ["technical questions", "behavioral questions"]
    assert scenario.known_facts == ["The scenario is an interview."]
    assert scenario.goal is None
    assert scenario.stakes is None


def test_raise_request_to_manager() -> None:
    scenario = parse_scenario("I need to ask my manager for a raise.")

    assert scenario.conversation_type == "workplace_conversation"
    assert scenario.user_role == "employee"
    assert scenario.counterpart_role == "manager"
    assert scenario.goal == "Discuss a raise or salary increase with the manager."
    assert "salary discussion" in scenario.topics
    assert scenario.stakes is None


def test_friend_cancelling_plans() -> None:
    scenario = parse_scenario("I want to talk to my friend because she keeps cancelling our plans.")

    assert scenario.conversation_type == "personal_conversation"
    assert scenario.user_role is None
    assert scenario.counterpart_role == "friend"
    assert scenario.goal == "Discuss the friend's repeated cancellations of plans."
    assert scenario.topics == ["cancelled plans"]


def test_minimal_job_interview_stays_limited() -> None:
    scenario = parse_scenario("Job interview.")

    assert scenario.conversation_type == "job_interview"
    assert scenario.user_role == "candidate"
    assert scenario.counterpart_role == "interviewer"
    assert scenario.goal is None
    assert scenario.topics == []
    assert scenario.stakes is None
    assert scenario.constraints == []


def test_unknown_scenario_preserves_text_and_adds_no_facts() -> None:
    raw = "I need to have a difficult conversation."

    scenario = parse_scenario(raw)

    assert scenario.raw_description == raw
    assert scenario.conversation_type is None
    assert scenario.user_role is None
    assert scenario.counterpart_role is None
    assert scenario.goal is None
    assert scenario.context is None
    assert scenario.topics == []
    assert scenario.known_facts == []


def test_blank_scenario_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        parse_scenario("  \n\t")