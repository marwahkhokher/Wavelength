"""
Persona & Scenario Schema — AI/Prompt Engineering

Persistent identity (Persona) and situation-specific context (Scenario) are
kept as separate objects, per the architecture review published 2026-08-20
("Dynamic Persona Architecture"). This supersedes the original schema.py,
which embedded scenario text and a frozen `initial_state` directly on the
persona - that shape couldn't survive persona reuse across multiple
situations, which is the core product requirement this module is built
against (PRD FR-PERS-4 through FR-PERS-11 cover generation/edit/finalize;
reuse across sessions is an explicit product requirement beyond the PRD).

Three-layer model:
  Persona            - who the other person stably is. Survives forever,
                        edited explicitly and rarely.
  Scenario           - what's happening right now and what the user wants.
                        One session's worth. Never modifies the persona.
  ConversationState  - how the persona feels right now, seeded from
                        Persona.baseline_dynamics + Scenario.difficulty,
                        mutated turn by turn. See conversation_state.py.

Design principle carried over from the original schema: traits are text,
dials are numbers. An LLM can act out "skeptical of unverified claims"
consistently; it cannot reproduce "skepticism: 0.73" consistently, and
nothing downstream does arithmetic on a trait. Numeric dials exist only on
`BaselineDynamics` (3 fields) and `ConversationState` (conversation_state.py)
- the only places code actually thresholds or increments a value.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid.uuid4())


class Mode(str, Enum):
    """Set on the Mode Selection screen (PRD Section 4.3 / FR-MODE-1)."""

    PROFESSIONAL = "professional"
    PERSONAL = "personal"


class DifficultyLevel(str, Enum):
    """
    PRD Section 9 Item #4 - difficulty levels aren't finalized yet.
    Lives on Scenario, not Persona: the same person can be practiced at
    different difficulties across different sessions, so difficulty can
    never be a fixed property of who they are.
    """

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ToneProfile(BaseModel):
    """
    Locked - derived from Persona.mode, not user-editable (see
    generation_prompt.py's TONE_RULES for the reviewable asset this is
    generated from). Letting a user turn "personal" speech into stiff
    corporate language would break the tone-realism requirement (PRD
    Section 6).
    """

    speech_register: str
    deflection_style: str
    example_phrase: str


class PersonaIdentity(BaseModel):
    name: str
    role_or_title: str
    relationship_to_user: str
    age_range: str  # e.g. "40-50" - never an exact age
    background: str  # 1-3 sentences, stable biography


class PersonaPersonality(BaseModel):
    """
    Kept deliberately narrow: the most reliable way to stop irrelevant
    generated content (a persona's favorite color, exact age, etc.) is to
    never have a field for it, not to filter it out at runtime.
    """

    traits: list[str] = Field(min_length=1, max_length=6)
    communication_style: list[str] = Field(min_length=1, max_length=6)
    values_and_priorities: list[str] = Field(default_factory=list, max_length=6)
    goals_in_conversation: list[str] = Field(default_factory=list, max_length=4)
    potential_triggers: list[str] = Field(default_factory=list, max_length=6)


class BaselineDynamics(BaseModel):
    """
    The ONLY numeric dials on the persistent persona. NOT a frozen session
    state - a seed that conversation_state.seed_dynamic_state() derives a
    fresh starting point from for every new conversation. Never read
    directly by the roleplay model; only ConversationState is.
    """

    patience: float = Field(0.5, ge=0.0, le=1.0)
    receptiveness: float = Field(0.5, ge=0.0, le=1.0)
    trust: float = Field(0.5, ge=0.0, le=1.0)


class KnownFact(BaseModel):
    """Small, explicit, provenanced - not a freeform memory dumping ground."""

    fact: str
    source: Literal["user_provided", "ai_inferred", "learned_in_session"]
    added_in_session_id: str | None = None


class Persona(BaseModel):
    persona_id: str = Field(default_factory=new_id)
    owner_user_id: str
    display_name: str  # user-facing label, e.g. "My CEO"
    version: int = 1  # bumped on every persistent edit; see PersonaVersionRecord
    schema_version: Literal["persona.v1"] = "persona.v1"

    identity: PersonaIdentity
    personality: PersonaPersonality
    tone: ToneProfile
    baseline_dynamics: BaselineDynamics = Field(default_factory=BaselineDynamics)
    known_facts: list[KnownFact] = Field(default_factory=list)
    behavioral_rules: list[str] = Field(default_factory=list, max_length=6)

    mode: Mode
    is_active: bool = True
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    # Fields explicitly editable by the user before/after finalizing (FR-PERS-5).
    # `tone` and `baseline_dynamics` are deliberately excluded - tone is
    # locked to mode, baseline_dynamics is engine-seeded. This is a product
    # decision beyond what FR-PERS-5 states outright and should carry
    # explicit sign-off, not be treated as already PRD-authorized.
    EDITABLE_FIELDS: list[str] = Field(
        default=["display_name", "identity", "personality", "behavioral_rules"],
        exclude=True,
    )


class PersonaVersionRecord(BaseModel):
    """
    Domain-level shape for an append-only version snapshot - what a
    `persona_versions` row should store. This module doesn't own storage;
    it defines the contract the backend/DB layer persists against.
    """

    persona_version_id: str = Field(default_factory=new_id)
    persona_id: str
    version: int
    snapshot: Persona
    changed_by: Literal["ai_generation", "user_edit", "system"]
    change_summary: str
    created_at: datetime = Field(default_factory=utcnow)


class Scenario(BaseModel):
    """
    One practice session's worth of "what's happening right now." Always
    generated fresh, even when reusing an existing persona - see
    scenario_generation.py. Never written back into the persona.
    """

    scenario_id: str = Field(default_factory=new_id)
    persona_id: str
    user_id: str

    situation_text_raw: str
    situation_summary: str
    user_goal: str
    interaction_type: Mode  # validated against persona.mode - see scenario_generation.py
    difficulty: DifficultyLevel
    duration_seconds: int = Field(gt=0)
    extracted_slots: dict[str, str] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=utcnow)
