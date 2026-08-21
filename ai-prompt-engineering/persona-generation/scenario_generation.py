"""
Scenario Generation — always generated fresh, even for a reused persona.

Takes situation_extraction's output plus (for the reuse flow) the existing
persona, and produces the Scenario-specific fields: a clean summary, the
user's goal, and any scenario-specific facts (extracted_slots) that are
relevant to THIS conversation only and are never written back into the
persona's known_facts.

Validates one thing specific to reuse (architecture doc Part 11): does the
situation's interaction_type match the persona's mode? A mismatch is
surfaced as an error rather than silently overriding either side - it
either means a UX mismatch to fix, or the user actually needs a different,
new persona.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from schema import DifficultyLevel, Mode, Persona, Scenario


class ScenarioMismatch(ValueError):
    """Situation's interaction_type doesn't match the target persona's mode."""


class GeneratedScenarioFields(BaseModel):
    situation_summary: str
    user_goal: str
    extracted_slots: dict[str, str] = Field(default_factory=dict)


_SYSTEM_TEMPLATE = """You are the scenario-generation stage of a persona-practice app \
(Confidence Building Platform). Given a raw situation description and, \
where available, who the other person is, produce:

- `situation_summary`: one clear sentence describing what's happening
- `user_goal`: one clear sentence describing what the user wants to \
accomplish in THIS conversation
- `extracted_slots`: a small dict of concrete, scenario-specific facts \
worth remembering for this conversation only (e.g. "tenure": "3 years", \
"deadline": "two weeks") - omit anything not explicitly stated. This is \
NOT persona information; nothing here describes who the other person is.

{persona_context}

## Output

Return ONLY a single JSON object matching this shape, no prose before or \
after it:

{output_schema}
"""

_PERSONA_CONTEXT = """The other person ({display_name}) is already established: \
{role_or_title}, traits: {traits}. Do not re-describe them - focus only on \
what's new in this situation."""

_NO_PERSONA_CONTEXT = "No established persona yet - describe the situation on its own."

_USER_TEMPLATE = """{situation_text}"""


def build_scenario_generation_messages(
    situation_text: str, persona: Persona | None
) -> list[dict[str, str]]:
    if not situation_text.strip():
        raise ValueError("situation_text must not be empty")

    persona_context = (
        _PERSONA_CONTEXT.format(
            display_name=persona.display_name,
            role_or_title=persona.identity.role_or_title,
            traits=", ".join(persona.personality.traits),
        )
        if persona is not None
        else _NO_PERSONA_CONTEXT
    )
    system = _SYSTEM_TEMPLATE.format(
        persona_context=persona_context,
        output_schema=json.dumps(GeneratedScenarioFields.model_json_schema(), indent=2),
    )
    user = _USER_TEMPLATE.format(situation_text=situation_text.strip())
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_generated_scenario(raw_response: str) -> GeneratedScenarioFields:
    return GeneratedScenarioFields.model_validate_json(raw_response)


def assemble_scenario(
    generated: GeneratedScenarioFields,
    *,
    situation_text_raw: str,
    persona: Persona,
    user_id: str,
    interaction_type: Mode,
    difficulty: DifficultyLevel,
    duration_seconds: int,
) -> Scenario:
    if interaction_type != persona.mode:
        raise ScenarioMismatch(
            f"Situation interaction_type={interaction_type.value!r} doesn't match "
            f"persona.mode={persona.mode.value!r} for persona {persona.persona_id} "
            f"({persona.display_name!r}). Either the user picked the wrong persona, "
            f"or this situation needs a different one."
        )
    return Scenario(
        persona_id=persona.persona_id,
        user_id=user_id,
        situation_text_raw=situation_text_raw,
        situation_summary=generated.situation_summary,
        user_goal=generated.user_goal,
        interaction_type=interaction_type,
        difficulty=difficulty,
        duration_seconds=duration_seconds,
        extracted_slots=generated.extracted_slots,
    )
