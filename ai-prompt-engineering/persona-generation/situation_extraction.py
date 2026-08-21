"""
Situation Extraction — free text in, structured SituationDraft out.

First LLM call in the pipeline. Deliberately also does the "useful tier"
gap scoring in the same call (see the architecture doc, Part 5 and Part 15):
splitting extraction and completeness-scoring into two calls would cost a
full network round trip for information the extraction call already has in
context. completeness.py then applies a purely deterministic gate on top of
this output - no second LLM call for that part at all.

When `known_persona` is passed (the reuse flow, architecture doc Part 10),
the persona's role/relationship are already known, so the model is told not
to re-derive or question them - only scenario-specific gaps can surface.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from schema import Mode, Persona


class InformationGap(BaseModel):
    """One thing the extractor noticed is missing or underspecified."""

    field: str  # e.g. "communication_style", "user_goal", "tenure"
    tier: Literal["required", "useful"]
    importance: int = Field(
        1, ge=1, le=5, description="Only meaningful for tier='useful'; ignored for 'required'."
    )
    question: str  # phrased, ready to show the user if this gap ends up being asked


class SituationDraft(BaseModel):
    interaction_type: Mode | None = None
    other_person_role: str | None = None
    relationship: str | None = None
    apparent_goal: str | None = None
    mentioned_facts: list[str] = Field(default_factory=list)
    gaps: list[InformationGap] = Field(default_factory=list)


_SYSTEM_TEMPLATE = """You are the first stage of a persona-generation pipeline for a mobile app \
where users practice real conversations with an AI playing a specific \
person (Confidence Building Platform, FR-PERS-1 through FR-PERS-4).

The user has described a situation, in their own words, possibly combined \
with a description of the person involved. Extract what's stated or \
strongly implied, and identify what's genuinely missing.

## What to extract

- `interaction_type`: "professional" or "personal" - infer from context if \
not explicit
- `other_person_role`: who the other person is (their role, or their \
relationship to the user)
- `relationship`: the user's relationship to that person
- `apparent_goal`: what the user is trying to accomplish in this specific \
conversation
- `mentioned_facts`: any other concrete facts stated about the person or \
situation, as short strings

{persona_context}

## Identifying gaps

Only two fields are ever REQUIRED: `other_person_role` and `apparent_goal` \
(or their equivalents, if a persona is already known - see above). If \
either is genuinely absent from the text (not just terse), add a gap with \
tier="required".

Beyond that, identify at most 3-4 USEFUL gaps - things that would \
materially change how realistic the generated persona or scenario is, and \
that cannot be safely defaulted. Score each 1-5 on how much it would \
actually change the outcome; do not inflate scores. Do NOT invent gaps for \
things that don't matter to generating a realistic persona (age, favorite \
things, physical description, etc.) - if it wouldn't change how the \
persona is generated, it is not a gap, it's just unknown and irrelevant.

Each gap needs a ready-to-ask `question`, phrased naturally, as if asking \
the user directly.

## Output

Return ONLY a single JSON object matching this shape, no prose before or \
after it:

{output_schema}
"""

_PERSONA_CONTEXT_KNOWN = """## An existing persona is already selected

This is: {display_name} ({role_or_title}, relationship: {relationship_to_user}).
Do NOT generate a gap for `other_person_role` or `relationship` - both are \
already known from this persona. Only look for gaps in the NEW situation \
itself (typically just `apparent_goal`, plus any situation-specific \
useful-tier facts)."""

_PERSONA_CONTEXT_NONE = """## No persona is selected yet

This is a new person the user hasn't described to the app before. \
`other_person_role` and `relationship` are both live candidates for \
required-tier gaps if the text doesn't establish them."""

_USER_TEMPLATE = """{raw_text}"""


def build_extraction_messages(
    raw_text: str, known_persona: Persona | None = None
) -> list[dict[str, str]]:
    if not raw_text.strip():
        raise ValueError("raw_text must not be empty")

    persona_context = (
        _PERSONA_CONTEXT_KNOWN.format(
            display_name=known_persona.display_name,
            role_or_title=known_persona.identity.role_or_title,
            relationship_to_user=known_persona.identity.relationship_to_user,
        )
        if known_persona is not None
        else _PERSONA_CONTEXT_NONE
    )
    system = _SYSTEM_TEMPLATE.format(
        persona_context=persona_context,
        output_schema=json.dumps(SituationDraft.model_json_schema(), indent=2),
    )
    user = _USER_TEMPLATE.format(raw_text=raw_text.strip())
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_situation_draft(raw_response: str) -> SituationDraft:
    return SituationDraft.model_validate_json(raw_response)
