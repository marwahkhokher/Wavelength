"""
Interactive live test against a real Gemini key - type an actual situation,
see an actual generated persona, then reuse it for a second situation.

Loads GEMINI_API_KEY from a local .env file (never from chat, never
hardcoded - see .env.example). If no key is set, this exits with a clear
message rather than silently falling back to canned data; use demo.py for
that (MockLLMClient) instead.

Simplification vs. the real product: the PRD has separate Scenario Input
and Persona Input screens (Section 5.5/5.6). This script asks for one
combined description and reuses it for both, since that's all that's
needed to exercise the pipeline - it is not a stand-in for the real UX.

Run: python try_it.py
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

import persona_editing as pe
import pipeline
from llm_client import GeminiLLMClient
from runtime_prompt import build_runtime_prompt
from schema import DifficultyLevel, Mode

MAX_QUESTION_ROUNDS = 3


def ask(prompt: str) -> str:
    return input(f"\n{prompt}\n> ").strip()


def print_persona(persona) -> None:
    print(f"\n{'=' * 70}\nPERSONA: {persona.display_name}\n{'=' * 70}")
    print(f"  Name: {persona.identity.name}")
    print(f"  Role: {persona.identity.role_or_title}")
    print(f"  Relationship: {persona.identity.relationship_to_user}")
    print(f"  Age range: {persona.identity.age_range}")
    print(f"  Background: {persona.identity.background}")
    print(f"  Traits: {', '.join(persona.personality.traits)}")
    print(f"  Communication style: {', '.join(persona.personality.communication_style)}")
    print(f"  Goals in conversation: {', '.join(persona.personality.goals_in_conversation)}")
    print(f"  Potential triggers: {', '.join(persona.personality.potential_triggers)}")
    print(f"  Speech register: {persona.tone.speech_register}")
    print(f"  Deflection style: {persona.tone.deflection_style}")
    print(f"  Example phrase: {persona.tone.example_phrase!r}")
    print(f"  Baseline dynamics: {persona.baseline_dynamics.model_dump()}")


async def gather_ready_situation(llm, mode: Mode, known_persona=None) -> str:
    """Loops the analyze -> ask -> answer cycle until status == ready,
    returning the full accumulated situation text."""
    text = ask(
        "Describe your situation"
        + (f" with {known_persona.display_name}" if known_persona else " (including who the other person is)")
        + ":"
    )
    asked_so_far = 0
    for _ in range(MAX_QUESTION_ROUNDS):
        draft, result = await pipeline.analyze_situation(
            llm, text, known_persona=known_persona, useful_questions_asked_so_far=asked_so_far
        )
        if result.status == "ready":
            print(f"\n[completeness: ready - extracted goal={draft.apparent_goal!r}]")
            return text
        print(f"\n[completeness: need_info - {len(result.questions)} question(s)]")
        for q in result.questions:
            answer = ask(q)
            text += f"\n{q} {answer}"
        asked_so_far += len(result.questions)
    return text


async def main() -> None:
    load_dotenv()
    if not os.environ.get("GEMINI_API_KEY"):
        print(
            "GEMINI_API_KEY is not set. Create a .env file in this directory with "
            "GEMINI_API_KEY=<your key> (see .env.example) and try again.\n"
            "For a no-key demo with scripted responses instead, run: python demo.py"
        )
        return

    llm = GeminiLLMClient()
    mode = Mode.PROFESSIONAL if ask("Mode - 'professional' or 'personal'?").lower().startswith("p") else Mode.PERSONAL
    display_name = ask("What should we call this persona? (e.g. 'My CEO')")

    situation_text = await gather_ready_situation(llm, mode)

    print("\nGenerating persona...")
    persona = await pipeline.create_persona(
        llm,
        owner_user_id="local-test-user",
        display_name=display_name,
        mode=mode,
        scenario_text=situation_text,
        persona_text=situation_text,
    )
    print_persona(persona)

    difficulty_input = ask("Difficulty for this conversation - easy / medium / hard?").lower()
    difficulty = {
        "easy": DifficultyLevel.EASY,
        "hard": DifficultyLevel.HARD,
    }.get(difficulty_input, DifficultyLevel.MEDIUM)

    scenario = await pipeline.create_scenario_for_persona(
        llm, persona=persona, situation_text=situation_text, user_id="local-test-user",
        difficulty=difficulty, duration_seconds=600,
    )
    state = pipeline.start_conversation(persona, scenario)
    print(f"\nScenario: {scenario.situation_summary}")
    print(f"Goal: {scenario.user_goal}")
    print(f"Seeded dynamic state: {state.model_dump()}")

    print("\n--- Runtime prompt (what the roleplay model would see) ---")
    print(build_runtime_prompt(persona, scenario, state))

    if ask("\nTry reusing this SAME persona for a different situation? (y/n)").lower().startswith("y"):
        new_situation = await gather_ready_situation(llm, mode, known_persona=persona)
        new_scenario = await pipeline.create_scenario_for_persona(
            llm, persona=persona, situation_text=new_situation, user_id="local-test-user",
            difficulty=difficulty, duration_seconds=600,
        )
        print(f"\nNew scenario, SAME persona (persona_id unchanged: "
              f"{new_scenario.persona_id == persona.persona_id}):")
        print(f"  {new_scenario.situation_summary}")
        print(f"  Traits, still unchanged: {persona.personality.traits}")

    edit_text = ask("\nTry editing the persona? Type an edit request, or leave blank to skip:")
    if edit_text:
        outcome = await pe.route_and_apply_edit(
            llm, persona=persona, current_difficulty=difficulty, edit_text=edit_text
        )
        if outcome.scope is pe.EditScope.PERMANENT:
            print(f"\n[permanent edit applied - version {persona.version} -> {outcome.persona.version}]")
            print_persona(outcome.persona)
        else:
            print(f"\n[session-level edit - difficulty {difficulty.value} -> {outcome.new_difficulty.value}, "
                  f"persona unchanged]")


if __name__ == "__main__":
    asyncio.run(main())
