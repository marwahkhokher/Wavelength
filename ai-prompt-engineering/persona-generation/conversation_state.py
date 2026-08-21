"""
Conversation State — hybrid LLM + deterministic (architecture doc Part 12).

The LLM's only job per turn is to classify the user's last message against
a FIXED label set (TurnClassification) - never to output a number directly.
Code owns 100% of the dial arithmetic via DIAL_DELTAS, a plain lookup
table. This is what makes the state auditable (the log is literally the
classification history) and prevents the "LLM invents a slightly different
number every time" drift problem numeric LLM output would otherwise cause.

Seeding (seed_dynamic_state) is the seam the original design was missing:
a persona's baseline_dynamics (schema.py) is a stable trait of the person,
but the ACTUAL starting point for one conversation also depends on how
hard this particular scenario is - the same CEO starts more guarded in a
"hard" salary negotiation than in an "easy" one.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from schema import BaselineDynamics, DifficultyLevel


class TurnPhase(str, Enum):
    OPENING = "opening"
    DISCUSSION = "discussion"
    NEGOTIATION = "negotiation"
    OBJECTION = "objection"
    PERSUASION = "persuasion"
    AGREEMENT = "agreement"
    REJECTION = "rejection"
    RESOLUTION = "resolution"


class TurnClassification(str, Enum):
    """The fixed label set the per-turn LLM call is constrained to output.

    This is deliberately a small closed enum, not free text - an LLM
    reliably picks one of five labels; it does not reliably produce a
    consistent numeric emotional score.
    """

    STRONG_ARGUMENT = "strong_argument"
    WEAK_ARGUMENT = "weak_argument"
    DISRESPECTFUL = "disrespectful"
    NEUTRAL = "neutral"
    CONCESSION_OFFERED = "concession_offered"


#: Deterministic dial deltas per classification. Code applies these -
#: never the LLM. Fields not mentioned for a given classification are
#: left unchanged.
DIAL_DELTAS: dict[TurnClassification, dict[str, float]] = {
    TurnClassification.STRONG_ARGUMENT: {"receptiveness": 0.10, "patience": 0.05},
    TurnClassification.WEAK_ARGUMENT: {"receptiveness": -0.05, "patience": -0.05},
    TurnClassification.DISRESPECTFUL: {
        "defensiveness": 0.20,
        "trust": -0.15,
        "patience": -0.10,
    },
    TurnClassification.NEUTRAL: {},
    TurnClassification.CONCESSION_OFFERED: {"trust": 0.10, "receptiveness": 0.05},
}

#: How much a harder scenario lowers the seeded starting dials, relative to
#: the persona's baseline_dynamics. Applied at conversation start only.
DIFFICULTY_PENALTY: dict[DifficultyLevel, float] = {
    DifficultyLevel.EASY: 0.0,
    DifficultyLevel.MEDIUM: 0.10,
    DifficultyLevel.HARD: 0.20,
}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


class DynamicState(BaseModel):
    """Per-conversation, mutable. Never persisted onto Persona (schema.py)."""

    receptiveness: float = Field(0.5, ge=0.0, le=1.0)
    patience: float = Field(0.5, ge=0.0, le=1.0)
    trust: float = Field(0.5, ge=0.0, le=1.0)
    defensiveness: float = Field(0.0, ge=0.0, le=1.0)
    turn_phase: TurnPhase = TurnPhase.OPENING


class StateLogEntry(BaseModel):
    """One row of the conversation's dynamic_state_log (schema.py's DB design, Part 9)."""

    turn: int
    classification: TurnClassification
    changes: dict[str, tuple[float, float]]  # field -> (before, after)


def seed_dynamic_state(baseline: BaselineDynamics, difficulty: DifficultyLevel) -> DynamicState:
    """Seed formula: persona.baseline_dynamics adjusted down by scenario difficulty.

    This runs once, at conversation start, and is the only place difficulty
    touches the dynamic state at all - it never re-applies mid-conversation.
    """
    penalty = DIFFICULTY_PENALTY[difficulty]
    return DynamicState(
        receptiveness=_clamp(baseline.receptiveness - penalty),
        patience=_clamp(baseline.patience - penalty),
        trust=_clamp(baseline.trust - penalty),
        defensiveness=0.0,
        turn_phase=TurnPhase.OPENING,
    )


def apply_turn_classification(
    state: DynamicState, classification: TurnClassification, turn: int
) -> tuple[DynamicState, StateLogEntry]:
    """Apply one turn's classification via the deterministic delta table.

    Returns a NEW DynamicState (does not mutate `state` in place) plus the
    log entry to append to the conversation's audit trail.
    """
    deltas = DIAL_DELTAS[classification]
    updated = state.model_copy()
    changes: dict[str, tuple[float, float]] = {}
    for field, delta in deltas.items():
        before = getattr(updated, field)
        after = _clamp(before + delta)
        setattr(updated, field, after)
        changes[field] = (before, after)
    return updated, StateLogEntry(turn=turn, classification=classification, changes=changes)


_CLASSIFICATION_SYSTEM_TEMPLATE = """You are classifying one user turn in a practice conversation, for a \
communication-coaching app. Read the user's latest message (in the context \
of the conversation so far) and classify it as EXACTLY ONE of:

- strong_argument: a specific, well-supported point (evidence, concrete \
examples, clear reasoning)
- weak_argument: a vague, unsupported, or purely emotional appeal
- disrespectful: dismissive, rude, or aggressive toward the other person
- concession_offered: the user offers a compromise or gives ground
- neutral: none of the above clearly applies (small talk, clarifying \
questions, etc.)

Output ONLY the single label, nothing else.
"""


def build_classification_messages(
    conversation_so_far: str, latest_user_message: str
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _CLASSIFICATION_SYSTEM_TEMPLATE},
        {
            "role": "user",
            "content": f"Conversation so far:\n{conversation_so_far}\n\n"
            f"Latest user message to classify:\n{latest_user_message}",
        },
    ]


def parse_classification(raw_response: str) -> TurnClassification:
    return TurnClassification(raw_response.strip().lower())
