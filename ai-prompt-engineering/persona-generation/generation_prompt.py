"""
Persona Generation Prompt — AI/Prompt Engineering

Turns (scenario_text, persona_text, mode) into the AI-generated subset of
Persona (schema.py): identity, personality, tone, and a first-pass
baseline_dynamics. This is generation step (1) of the pipeline - see
pipeline.py for how it chains with situation extraction, completeness
checking, and validation.

Scope note, unchanged from the original version of this module: this step
does NOT receive difficulty. Per the application flow (PRD Section 4.3),
difficulty is chosen after the profile is generated and shown to the user,
and it now lives on Scenario (schema.py), not Persona at all - the same
person is practiced at different difficulties across different sessions.

Tone rules are kept as a plain, reviewable data structure (TONE_RULES
below) rather than folded into prose inside the prompt string, per
FR-TONE-2 ("tone rules...maintained as a defined, reviewable asset rather
than buried in code, since tone is a product-quality surface that will be
iterated on").
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

# NOTE: this directory is `persona-generation/` (hyphenated), which cannot be
# imported as a real Python package (`import persona-generation` is a syntax
# error, and no dotted path can reach it) - so this is a flat sibling-module
# import, not `from .schema import ...`. Callers need this directory on
# sys.path directly (see tests/conftest.py).
from schema import (
    BaselineDynamics,
    Mode,
    Persona,
    PersonaIdentity,
    PersonaPersonality,
    ToneProfile,
)


class ToneRule(BaseModel):
    """One mode's tone rules - the reviewable asset FR-TONE-2 asks for."""

    register_style: str
    directness: str
    deflection_rule: str
    example_phrases: list[str]
    language_note: str


TONE_RULES: dict[Mode, ToneRule] = {
    Mode.PROFESSIONAL: ToneRule(
        register_style="Measured, polite, face-saving. Workplace norms throughout.",
        directness="Diplomatic. Avoids blunt confrontation even under pressure.",
        deflection_rule=(
            "When the persona doesn't know something or is put on the spot, "
            "they deflect diplomatically rather than admitting it bluntly - "
            "buy time or redirect, never a flat 'I don't know.'"
        ),
        example_phrases=[
            "Great question, I'll get back to you.",
            "Let me loop back on that and circle round.",
            "I hear you - let's park that and come back to it.",
        ],
        language_note="Standard professional English. No slang, no code-switching.",
    ),
    Mode.PERSONAL: ToneRule(
        register_style="Direct and casual - the way people talk to friends and family.",
        directness="Blunt and unpolished. Says what they think without softening it.",
        deflection_rule=(
            "When the persona doesn't know something, they say so bluntly and "
            "unpolished - no corporate hedging."
        ),
        example_phrases=[
            "Yaar, I don't know.",
            "Honestly? No idea.",
            "I don't know man, don't ask me that.",
        ],
        language_note=(
            "Everyday colloquial language, including natural code-switching / "
            "mixed-language phrasing where it fits the persona (PRD Section 9 "
            "Item #7 - full language support is unconfirmed; keep this light "
            "rather than committing to a specific non-English language)."
        ),
    ),
}


class GeneratedPersonaFields(BaseModel):
    """
    The LLM's actual output contract for this generation step - reuses
    Persona's own nested models directly rather than duplicating a flat
    field list, so there is exactly one definition of what "identity" or
    "personality" means.

    Deliberately excludes everything else on Persona:
      - persona_id / owner_user_id / display_name / version: app-assigned,
        not generated
      - mode: caller already knows it, passed in unchanged, not generated
      - known_facts / behavioral_rules: start empty, added later via
        explicit user edits or learned facts, not invented at generation time
      - is_finalized-equivalent lifecycle fields: app-owned
    """

    identity: PersonaIdentity
    personality: PersonaPersonality
    tone: ToneProfile
    baseline_dynamics: BaselineDynamics = Field(default_factory=BaselineDynamics)


_SYSTEM_TEMPLATE = """You are building a persona profile for a mobile app that lets users \
practice real conversations with an AI that plays a specific person \
(Confidence Building Platform, PRD Section 5.6 / FR-PERS-4).

The user has described a situation they need to handle, and separately \
described the person they need to handle it with. Your job is to turn that \
description into one coherent, specific person - not a generic assistant, \
and not a caricature.

## Register for this session: {mode_value}

{register_style}
Directness: {directness}
{deflection_rule}
Language: {language_note}

Example phrases in this register (for calibration only - do not reuse them \
verbatim unless they genuinely fit):
{example_phrases}

This register must show up in `tone.speech_register`, `tone.deflection_style`, \
and `tone.example_phrase` in your output, and should be reflected implicitly \
in `personality.communication_style` too. It is about HOW this person talks, \
not about how difficult or easy they are to deal with - difficulty is set \
separately, later, by the user, and you have no information about it. Do \
not infer or imply a difficulty level anywhere in your output.

## What to infer

From the scenario and the persona description together, infer:
- Who this person specifically is (name; an age range, not an exact age; \
their role or relationship to the user)
- A short, grounded background (1-3 sentences) that explains why they'd act \
this way in this scenario
- 3-6 personality traits and 3-6 communication-style phrases consistent \
with that background - short phrases, not paragraphs, and not an \
exhaustive list
- What this person actually wants out of THIS conversation (their goals, \
not the user's)
- What tends to make them defensive, frustrated, or shut down (potential \
triggers) - grounded in the scenario, not generic

Stay strictly inside what the scenario and persona description imply or \
directly state. Do not invent unrelated backstory. If the persona \
description is thin, keep inferred details plausible and minimal rather \
than elaborate.

## baseline_dynamics

Set `patience`, `receptiveness`, and `trust` (each 0.0-1.0, default 0.5) \
based ONLY on the personality you just inferred - e.g. someone described as \
"impatient" or "no-nonsense" should get a lower starting `patience`, \
someone described as "warm" or "easygoing" a higher `receptiveness`. Do \
NOT factor in difficulty - you don't have it. Small adjustments only \
(typically within 0.15 of 0.5); this is a starting point for a dynamic \
system, not a verdict on the person.

## Output

Return ONLY a single JSON object matching this shape, no prose before or \
after it:

{output_schema}
"""

_USER_TEMPLATE = """Scenario:
{scenario_text}

Persona description:
{persona_text}
"""


def build_generation_messages(
    scenario_text: str, persona_text: str, mode: Mode
) -> list[dict[str, str]]:
    """Build the chat messages for one persona-generation call.

    Returns [{"role": "system", ...}, {"role": "user", ...}] - the shape
    most chat-completion APIs expect. Callers using a single-prompt API can
    just concatenate the two contents.
    """
    if not scenario_text.strip():
        raise ValueError("scenario_text must not be empty")
    if not persona_text.strip():
        raise ValueError("persona_text must not be empty")

    rule = TONE_RULES[mode]
    system = _SYSTEM_TEMPLATE.format(
        mode_value=mode.value,
        register_style=rule.register_style,
        directness=rule.directness,
        deflection_rule=rule.deflection_rule,
        language_note=rule.language_note,
        example_phrases="\n".join(f"- {p}" for p in rule.example_phrases),
        output_schema=json.dumps(GeneratedPersonaFields.model_json_schema(), indent=2),
    )
    user = _USER_TEMPLATE.format(
        scenario_text=scenario_text.strip(), persona_text=persona_text.strip()
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_generated_fields(raw_response: str) -> GeneratedPersonaFields:
    """Validate a raw LLM JSON response against the generation contract."""
    return GeneratedPersonaFields.model_validate_json(raw_response)


def assemble_persona(
    generated: GeneratedPersonaFields,
    *,
    owner_user_id: str,
    display_name: str,
    mode: Mode,
) -> Persona:
    """Combine this step's output with what the app already knows.

    No `difficulty` parameter (unlike the previous version of this
    function) - difficulty doesn't touch Persona at all any more, it lives
    on Scenario (schema.py), set later by scenario_generation.py.
    """
    return Persona(
        owner_user_id=owner_user_id,
        display_name=display_name,
        identity=generated.identity,
        personality=generated.personality,
        tone=generated.tone,
        baseline_dynamics=generated.baseline_dynamics,
        mode=mode,
    )
