"""
End-to-end pipeline tests using MockLLMClient - exercises the exact two
flows from the architecture doc (Part 4: first contact, Part 10: reuse)
without any live LLM call, matching how voice-tech-infra's MockAIServiceClient
was used to validate that pipeline end to end.
"""

import pytest

import pipeline
from llm_client import MockLLMClient
from schema import DifficultyLevel, Mode

GENERATED_PERSONA_JSON = """{
  "identity": {
    "name": "Dana Whitfield",
    "role_or_title": "Chief Executive Officer",
    "relationship_to_user": "employer",
    "age_range": "40-50",
    "background": "Has led the company for six years and prides herself on being fair, but budget conversations make her visibly uncomfortable."
  },
  "personality": {
    "traits": ["direct", "no-nonsense", "results-oriented"],
    "communication_style": ["short sentences", "redirects to process"],
    "values_and_priorities": ["predictability"],
    "goals_in_conversation": ["keep the conversation on schedule"],
    "potential_triggers": ["being asked for a firm number on the spot"]
  },
  "tone": {
    "speech_register": "measured, polite, face-saving",
    "deflection_style": "promises to follow up rather than answering directly",
    "example_phrase": "Great question, I'll get back to you."
  },
  "baseline_dynamics": {"patience": 0.4, "receptiveness": 0.5, "trust": 0.5}
}"""

SITUATION_DRAFT_READY_JSON = """{
  "interaction_type": "professional",
  "other_person_role": "CEO",
  "relationship": "employee-employer",
  "apparent_goal": "salary increase",
  "mentioned_facts": ["three years tenure", "direct communication style"],
  "gaps": []
}"""

SITUATION_DRAFT_NEEDS_INFO_JSON = """{
  "interaction_type": "professional",
  "other_person_role": "CEO",
  "relationship": "employee-employer",
  "apparent_goal": "salary increase",
  "mentioned_facts": [],
  "gaps": [
    {"field": "communication_style", "tier": "useful", "importance": 4,
     "question": "How would you describe your CEO's communication style?"}
  ]
}"""

GENERATED_SCENARIO_JSON = """{
  "situation_summary": "Requesting approval for a one-month leave",
  "user_goal": "Get Dana to approve one month of leave",
  "extracted_slots": {"duration": "one month"}
}"""


async def test_analyze_situation_ready_needs_no_questions():
    llm = MockLLMClient(responses=[SITUATION_DRAFT_READY_JSON])
    draft, result = await pipeline.analyze_situation(llm, "I have a meeting with my CEO about a raise.")
    assert result.status == "ready"
    assert result.questions == []
    assert draft.other_person_role == "CEO"


async def test_analyze_situation_returns_questions_when_gaps_present():
    llm = MockLLMClient(responses=[SITUATION_DRAFT_NEEDS_INFO_JSON])
    draft, result = await pipeline.analyze_situation(llm, "I have a meeting with my CEO.")
    assert result.status == "need_info"
    assert len(result.questions) == 1
    assert "communication style" in result.questions[0]


async def test_first_contact_flow_creates_a_valid_persona():
    """Architecture doc Part 4, condensed: analyze -> ready -> generate persona."""
    llm = MockLLMClient(responses=[SITUATION_DRAFT_READY_JSON, GENERATED_PERSONA_JSON])

    draft, result = await pipeline.analyze_situation(
        llm, "I have a meeting with my CEO and want a raise. He's direct, three years tenure."
    )
    assert result.status == "ready"

    persona = await pipeline.create_persona(
        llm,
        owner_user_id="user-1",
        display_name="My CEO",
        mode=Mode.PROFESSIONAL,
        scenario_text="Salary negotiation with my CEO",
        persona_text="My CEO Dana, direct, three years I've known her",
    )
    assert persona.identity.name == "Dana Whitfield"
    assert persona.mode == Mode.PROFESSIONAL
    assert persona.baseline_dynamics.patience == 0.4
    assert len(llm.calls) == 2


async def test_reuse_flow_skips_persona_generation_entirely():
    """Architecture doc Part 10: existing persona + new situation -> new Scenario only."""
    llm = MockLLMClient(
        responses=[SITUATION_DRAFT_READY_JSON, GENERATED_PERSONA_JSON, GENERATED_SCENARIO_JSON]
    )
    # Build the persona once, as if it already existed from a prior session.
    draft, result = await pipeline.analyze_situation(llm, "meeting my CEO")
    persona = await pipeline.create_persona(
        llm, owner_user_id="user-1", display_name="My CEO", mode=Mode.PROFESSIONAL,
        scenario_text="salary talk", persona_text="my CEO Dana",
    )
    assert len(llm.calls) == 2

    # Now reuse it for a brand-new situation - only ONE more call (scenario
    # generation), never a second persona-generation call.
    scenario = await pipeline.create_scenario_for_persona(
        llm,
        persona=persona,
        situation_text="I need to request a one-month leave.",
        user_id="user-1",
        difficulty=DifficultyLevel.HARD,
        duration_seconds=600,
    )
    assert len(llm.calls) == 3  # not 4 - persona generation was never re-invoked
    assert scenario.persona_id == persona.persona_id
    assert scenario.situation_summary == "Requesting approval for a one-month leave"
    assert scenario.interaction_type == persona.mode


async def test_reused_persona_keeps_its_personality_across_scenarios():
    """The CEO must behave consistently whether it's a salary or a leave conversation."""
    llm = MockLLMClient(responses=[GENERATED_PERSONA_JSON])
    persona = await pipeline.create_persona(
        llm, owner_user_id="user-1", display_name="My CEO", mode=Mode.PROFESSIONAL,
        scenario_text="salary talk", persona_text="my CEO Dana",
    )
    original_traits = list(persona.personality.traits)

    llm2 = MockLLMClient(responses=[GENERATED_SCENARIO_JSON])
    scenario = await pipeline.create_scenario_for_persona(
        llm2, persona=persona, situation_text="leave request", user_id="user-1",
        difficulty=DifficultyLevel.EASY, duration_seconds=300,
    )
    # persona object is untouched by generating a new scenario for it
    assert persona.personality.traits == original_traits
    assert scenario.persona_id == persona.persona_id


async def test_start_conversation_seeds_state_from_persona_and_scenario_difficulty():
    llm = MockLLMClient(responses=[GENERATED_PERSONA_JSON, GENERATED_SCENARIO_JSON])
    persona = await pipeline.create_persona(
        llm, owner_user_id="user-1", display_name="My CEO", mode=Mode.PROFESSIONAL,
        scenario_text="s", persona_text="p",
    )
    scenario = await pipeline.create_scenario_for_persona(
        llm, persona=persona, situation_text="leave request", user_id="user-1",
        difficulty=DifficultyLevel.HARD, duration_seconds=300,
    )
    state = pipeline.start_conversation(persona, scenario)
    # baseline patience 0.4, HARD penalty 0.2
    assert state.patience == pytest.approx(0.2)


async def test_persona_generation_failure_raises_with_issue_detail():
    bad_persona_json = GENERATED_PERSONA_JSON.replace(
        '"background": "Has led the company for six years and prides herself on being fair, but budget conversations make her visibly uncomfortable."',
        '"background": ""',
    )
    llm = MockLLMClient(responses=[bad_persona_json])
    with pytest.raises(pipeline.PersonaGenerationFailed) as excinfo:
        await pipeline.create_persona(
            llm, owner_user_id="user-1", display_name="My CEO", mode=Mode.PROFESSIONAL,
            scenario_text="s", persona_text="p",
        )
    assert "background" in str(excinfo.value).lower()
