"""
Standalone demo, not a test: walks through both product flows from the
architecture doc using MockLLMClient (no live API key needed/used) so the
wiring can be seen working end to end.

Run: python demo.py
"""

from __future__ import annotations

import asyncio

import pipeline
from conversation_state import TurnClassification, apply_turn_classification
from llm_client import MockLLMClient
from runtime_prompt import build_runtime_prompt
from schema import DifficultyLevel, Mode

CEO_PERSONA_JSON = """{
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

SITUATION_READY_JSON = """{
  "interaction_type": "professional", "other_person_role": "CEO",
  "relationship": "employee-employer", "apparent_goal": "salary increase",
  "mentioned_facts": ["three years tenure", "direct communication style"], "gaps": []
}"""

LEAVE_SCENARIO_JSON = """{
  "situation_summary": "Requesting approval for a one-month leave",
  "user_goal": "Get Dana to approve one month of leave",
  "extracted_slots": {"duration": "one month"}
}"""


async def main() -> None:
    print("=" * 78)
    print("SESSION 1 — first contact: \"I have a meeting with my CEO and want a raise\"")
    print("=" * 78)

    llm = MockLLMClient(responses=[SITUATION_READY_JSON, CEO_PERSONA_JSON])
    draft, completeness = await pipeline.analyze_situation(
        llm, "I have a meeting with my CEO and want to convince him to increase my salary. "
             "He's pretty direct, I've been here three years."
    )
    print(f"\nExtracted: role={draft.other_person_role!r} goal={draft.apparent_goal!r}")
    print(f"Completeness: {completeness.status} ({len(completeness.questions)} question(s))")

    ceo = await pipeline.create_persona(
        llm, owner_user_id="user-1", display_name="My CEO", mode=Mode.PROFESSIONAL,
        scenario_text="Salary negotiation with my CEO",
        persona_text="My CEO Dana, direct, I've worked with her three years",
    )
    print(f"\nPersona created: {ceo.display_name!r}")
    print(f"  traits: {ceo.personality.traits}")
    print(f"  baseline_dynamics: {ceo.baseline_dynamics.model_dump()}")
    print(f"  LLM calls so far: {len(llm.calls)}")

    print("\n" + "=" * 78)
    print("SESSION 2 — months later: \"I need to request a one-month leave\"")
    print("User selects the SAME persona: \"My CEO\"")
    print("=" * 78)

    llm2 = MockLLMClient(responses=[LEAVE_SCENARIO_JSON])
    scenario = await pipeline.create_scenario_for_persona(
        llm2, persona=ceo, situation_text="I need to request a one-month leave.",
        user_id="user-1", difficulty=DifficultyLevel.HARD, duration_seconds=600,
    )
    print(f"\nScenario created (persona NOT regenerated - {len(llm2.calls)} LLM call, "
          f"not the 2 the first session needed):")
    print(f"  situation_summary: {scenario.situation_summary!r}")
    print(f"  same persona_id: {scenario.persona_id == ceo.persona_id}")
    print(f"  same traits, unchanged: {ceo.personality.traits}")

    state = pipeline.start_conversation(ceo, scenario)
    print(f"\nDynamic state seeded (baseline patience=0.4, HARD scenario -0.2 penalty):")
    print(f"  {state.model_dump()}")

    print("\n--- Runtime prompt for turn 1 (this is what the roleplay LLM actually sees) ---")
    prompt = build_runtime_prompt(ceo, scenario, state)
    print(prompt)

    print("\n--- User gives a weak, vague argument ('I just really need the time off') ---")
    state, log = apply_turn_classification(state, TurnClassification.WEAK_ARGUMENT, turn=1)
    print(f"State change: {log.changes}")

    print("\n--- User then gets disrespectful ('this is a joke, you never approve anything') ---")
    state, log = apply_turn_classification(state, TurnClassification.DISRESPECTFUL, turn=2)
    print(f"State change: {log.changes}")
    print(f"\nState after 2 turns: {state.model_dump()}")

    print("\n--- Runtime prompt is rebuilt fresh for turn 3, reflecting the new state ---")
    prompt_turn_3 = build_runtime_prompt(ceo, scenario, state)
    for line in prompt_turn_3.splitlines():
        if "Patience" in line or "Trust" in line or "Defensiveness" in line:
            print(f"  {line}")
    print("\n(full persona identity/traits/rules are unchanged and re-injected every turn - ")
    print(" this is what keeps the persona from drifting into a generic chatbot, per")
    print(" the architecture doc's Part 13.)")


if __name__ == "__main__":
    asyncio.run(main())
