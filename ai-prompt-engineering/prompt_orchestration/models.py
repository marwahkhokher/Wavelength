"""Internal models for the Conversation Orchestrator (Taha's ownership).

These models represent Person 5's internal state and are NOT part of the
shared voice-tech-infra contracts. They track conversation state, persona
dynamics, and structured session memory.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Session State Machine
# ---------------------------------------------------------------------------

class SessionState(str, Enum):
    """High-level session lifecycle states."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    ENDING = "ending"
    COMPLETED = "completed"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Conversation Messages
# ---------------------------------------------------------------------------

class ConversationMessage(BaseModel):
    """A single message in the conversation history."""
    role: Literal["user", "assistant", "system"]
    content: str
    turn_id: int
    timestamp: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Structured Session Memory
# ---------------------------------------------------------------------------

class StructuredMemory(BaseModel):
    """Extracted key facts from the conversation for long-term context."""
    user_claims: list[str] = Field(default_factory=list)
    key_topics_discussed: list[str] = Field(default_factory=list)
    unresolved_topics: list[str] = Field(default_factory=list)
    user_strengths_observed: list[str] = Field(default_factory=list)
    user_weaknesses_observed: list[str] = Field(default_factory=list)
    important_facts: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Persona State (Mutable)
# ---------------------------------------------------------------------------

class PersonaState(BaseModel):
    """Mutable persona state that shifts based on user performance.

    The core persona traits (from SessionContext.persona) are immutable.
    These values represent the persona's *current conversational disposition*.
    """
    receptiveness: float = Field(default=0.5, ge=0.0, le=1.0)
    defensiveness: float = Field(default=0.2, ge=0.0, le=1.0)
    interest_level: float = Field(default=0.5, ge=0.0, le=1.0)
    satisfaction: float = Field(default=0.5, ge=0.0, le=1.0)
    pressure_level: float = Field(default=0.3, ge=0.0, le=1.0)

    def adjust_from_scores(
        self,
        confidence_score: float | None = None,
        clarity_score: float | None = None,
        relevance_score: float | None = None,
    ) -> None:
        """Shift persona state based on evaluation scores (0-100 scale)."""
        if confidence_score is not None:
            # Strong confidence → persona becomes more receptive
            if confidence_score > 75:
                self.receptiveness = min(1.0, self.receptiveness + 0.05)
                self.interest_level = min(1.0, self.interest_level + 0.03)
            elif confidence_score < 40:
                self.pressure_level = min(1.0, self.pressure_level + 0.05)

        if clarity_score is not None:
            if clarity_score > 80:
                self.satisfaction = min(1.0, self.satisfaction + 0.05)
            elif clarity_score < 50:
                self.defensiveness = min(1.0, self.defensiveness + 0.03)

        if relevance_score is not None:
            if relevance_score > 85:
                self.interest_level = min(1.0, self.interest_level + 0.05)
            elif relevance_score < 50:
                self.interest_level = max(0.0, self.interest_level - 0.05)


# ---------------------------------------------------------------------------
# Conversation State (Per-Session)
# ---------------------------------------------------------------------------

class ConversationState(BaseModel):
    """Complete state for one active conversation session."""
    session_id: str
    user_id: str
    state: SessionState = SessionState.INITIALIZING
    current_turn: int = 0
    conversation_history: list[ConversationMessage] = Field(default_factory=list)
    structured_memory: StructuredMemory = Field(default_factory=StructuredMemory)
    persona_state: PersonaState = Field(default_factory=PersonaState)
    history_summary: str | None = None
    created_at: float = Field(default_factory=time.time)
    last_activity: float = Field(default_factory=time.time)

    # Analysis cache for the most recent turn
    last_analysis: dict[str, Any] | None = None

    def add_message(self, role: Literal["user", "assistant", "system"], content: str) -> ConversationMessage:
        """Add a message and increment turn counter for user messages."""
        if role == "user":
            self.current_turn += 1
        msg = ConversationMessage(
            role=role,
            content=content,
            turn_id=self.current_turn,
        )
        self.conversation_history.append(msg)
        self.last_activity = time.time()
        return msg

    def get_recent_messages(self, n: int = 6) -> list[ConversationMessage]:
        """Return the most recent n messages."""
        return self.conversation_history[-n:]
