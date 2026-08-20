"""
Persona Profile Schema — AI/Prompt Engineering

Defines the full shape of a generated persona, per PRD FR-PERS-4 through
FR-PERS-11. This is the object that:
  - the AI generates from (scenario_text + persona_text + mode)
  - the user edits on the Persona Profile screen (Part B)
  - gets "finalized" and locked before the session starts (FR-PERS-6)
  - drives the live session (FR-SESS-6) and dynamic-engine's turn-by-turn
    state shifts

Design principle: every field is tagged as editable or locked. Locked
fields exist so the persona can't be edited into something incoherent
(e.g. a user shouldn't be able to hand-edit their own hidden receptiveness
score to make the session trivially easy — that value is earned through
the conversation, not set by the user).
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Mode(str, Enum):
    """Set on the Mode Selection screen (Section 4.3), passed in unchanged."""
    PROFESSIONAL = "professional"
    PERSONAL = "personal"


class DifficultyLevel(str, Enum):
    """
    PRD Item #4 (Section 9) — difficulty levels aren't finalized yet.
    Using a 3-tier placeholder so persona-generation isn't blocked;
    swap this enum out once Product/Research confirms the real levels
    and what concretely changes at each one.
    """
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class PersonaState(BaseModel):
    """
    Hidden state that the dynamic-engine (turn-by-turn) reads and updates.
    Starting values are set here at generation time, based on difficulty
    and the scenario's described tension. The user never sees or edits
    these directly (Section 7.4 of the earlier PRD: "showing the actual
    rubric mid-conversation would kill the immersion").
    """
    receptiveness: float = Field(0.5, ge=0.0, le=1.0)
    patience: float = Field(0.5, ge=0.0, le=1.0)
    trust: float = Field(0.5, ge=0.0, le=1.0)

    def is_valid(self) -> bool:
        return all(0.0 <= v <= 1.0 for v in (self.receptiveness, self.patience, self.trust))


class ToneProfile(BaseModel):
    """
    Section 6 — Professional vs Personal is a REGISTER difference, not a
    difficulty difference. These fields let the generation prompt encode
    that distinction concretely instead of leaving it to the model's
    defaults (Section 6.1: "has to be built deliberately").
    Locked after generation — tone is derived from `mode`, not user-editable,
    since letting a user turn "personal" speech into stiff corporate language
    (or vice versa) would break the tone-realism requirement.
    """
    speech_register: str  # e.g. "measured, polite, face-saving" vs "direct, casual, unpolished"
    deflection_style: str  # how they avoid admitting they don't know something
    example_phrase: str  # one grounding example, e.g. "Great question, I'll get back to you."


class PersonaProfile(BaseModel):
    # --- Identity (AI-generated, user-editable) ---
    name: str
    age_range: str  # e.g. "30-40", kept as a range rather than exact age
    role_or_relationship: str  # e.g. "Direct manager", "Older sibling"
    background: str  # 1-3 sentences of context AI inferred from persona_text

    # --- Personality (AI-generated, user-editable) ---
    personality_traits: list[str]  # e.g. ["direct", "impatient", "detail-oriented"]
    communication_style: list[str]  # e.g. ["short sentences", "asks pointed questions"]
    goals_in_conversation: list[str]  # what THIS persona wants out of the exchange
    potential_triggers: list[str]  # what tends to make them defensive/frustrated

    # --- Tone (AI-generated, LOCKED — derived from mode, not user-editable) ---
    tone: ToneProfile

    # --- Session config (set by user on the same screen, Part C) ---
    mode: Mode
    difficulty: DifficultyLevel

    # --- Hidden state (AI-generated at creation, LOCKED from user, engine-owned during session) ---
    initial_state: PersonaState = Field(default_factory=PersonaState)

    # --- Provenance / lifecycle ---
    generated_from_scenario: str  # raw scenario_text, stored for traceability
    generated_from_persona_description: str  # raw persona_text
    is_finalized: bool = False  # FR-PERS-6: must be true before session can begin

    # Fields explicitly editable by the user before finalizing.
    # Used by the frontend to know which inputs to render as editable
    # vs. read-only, and by the backend to reject edits to locked fields.
    EDITABLE_FIELDS: list[str] = Field(
        default=[
            "name",
            "age_range",
            "role_or_relationship",
            "background",
            "personality_traits",
            "communication_style",
            "goals_in_conversation",
            "potential_triggers",
        ],
        exclude=True,  # this is metadata about the schema, not a stored field
    )
