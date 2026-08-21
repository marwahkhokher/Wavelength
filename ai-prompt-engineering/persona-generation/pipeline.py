"""
Pipeline — plain-Python orchestration (architecture doc Part 16: no agent
framework - this pipeline is a mostly-linear sequence with one bounded
loop, exactly the case where LangGraph/LangChain overhead doesn't buy
anything a handful of async functions and Pydantic models don't already
give you more legibly).

Wires the modules together into the two flows the product needs:
  - analyze_situation()      : shared first step for both flows
  - create_persona()         : first-contact flow (architecture doc Part 4)
  - create_scenario_for_persona() : reuse flow (architecture doc Part 10)
  - start_conversation()     : seeds dynamic state for either flow

This module owns no storage. It's what a backend endpoint calls; callers
pass in an LLMClient and get back plain domain objects (schema.py) to
persist however their DB layer is built - see the architecture doc Part 9
for the schema this is meant to be persisted against.
"""

from __future__ import annotations

from completeness import CompletenessResult, check_completeness
from conversation_state import DynamicState, seed_dynamic_state
from generation_prompt import assemble_persona, build_generation_messages, parse_generated_fields
from llm_client import LLMClient
from schema import DifficultyLevel, Mode, Persona, Scenario
from scenario_generation import (
    assemble_scenario,
    build_scenario_generation_messages,
    parse_generated_scenario,
)
from situation_extraction import (
    SituationDraft,
    build_extraction_messages,
    parse_situation_draft,
)
from validation import ValidationResult, validate_deterministic


class PersonaGenerationFailed(RuntimeError):
    """Raised when a generated persona fails deterministic validation."""

    def __init__(self, result: ValidationResult) -> None:
        self.result = result
        messages = "; ".join(i.message for i in result.issues if i.severity == "error")
        super().__init__(f"Generated persona failed validation: {messages}")


async def analyze_situation(
    llm: LLMClient,
    raw_text: str,
    *,
    known_persona: Persona | None = None,
    useful_questions_asked_so_far: int = 0,
) -> tuple[SituationDraft, CompletenessResult]:
    """Shared first step for both flows - architecture doc Part 15, functions 1-3 merged.

    One LLM call (extraction + useful-tier gap scoring) followed by a
    purely deterministic completeness gate. Callers loop this: if
    `status == "need_info"`, show `questions` to the user, fold their
    answers into `raw_text`, and call again with an incremented
    `useful_questions_asked_so_far`.
    """
    messages = build_extraction_messages(raw_text, known_persona=known_persona)
    raw = await llm.complete(messages)
    draft = parse_situation_draft(raw)
    result = check_completeness(draft, useful_questions_asked_so_far=useful_questions_asked_so_far)
    return draft, result


async def create_persona(
    llm: LLMClient,
    *,
    owner_user_id: str,
    display_name: str,
    mode: Mode,
    scenario_text: str,
    persona_text: str,
) -> Persona:
    """First-contact flow (architecture doc Part 4).

    Call only once `analyze_situation()` has returned status="ready" -
    this function does not itself check completeness.
    """
    messages = build_generation_messages(scenario_text, persona_text, mode)
    raw = await llm.complete(messages)
    generated = parse_generated_fields(raw)

    check = validate_deterministic(generated)
    if not check.passed:
        raise PersonaGenerationFailed(check)

    return assemble_persona(
        generated, owner_user_id=owner_user_id, display_name=display_name, mode=mode
    )


async def create_scenario_for_persona(
    llm: LLMClient,
    *,
    persona: Persona,
    situation_text: str,
    user_id: str,
    difficulty: DifficultyLevel,
    duration_seconds: int,
) -> Scenario:
    """Reuse flow (architecture doc Part 10): persona already exists and is
    unchanged; only a fresh Scenario is generated and attached to it.
    """
    messages = build_scenario_generation_messages(situation_text, persona)
    raw = await llm.complete(messages)
    generated = parse_generated_scenario(raw)
    return assemble_scenario(
        generated,
        situation_text_raw=situation_text,
        persona=persona,
        user_id=user_id,
        interaction_type=persona.mode,
        difficulty=difficulty,
        duration_seconds=duration_seconds,
    )


def start_conversation(persona: Persona, scenario: Scenario) -> DynamicState:
    """Seeds dynamic state at session start - architecture doc Part 12's seed formula."""
    return seed_dynamic_state(persona.baseline_dynamics, scenario.difficulty)
