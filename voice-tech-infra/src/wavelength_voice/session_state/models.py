"""Data model for a live (or recently-disconnected) conversation session."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SessionStatus(str, Enum):
    #: A websocket connection is attached and live.
    ACTIVE = "active"
    #: No connection attached; within the reconnect grace window.
    DISCONNECTED_GRACE = "disconnected_grace"
    #: Grace window elapsed with no reconnect - session is dead.
    EXPIRED = "expired"
    #: Ended deliberately (user finished, or AI service signalled end_session).
    ENDED = "ended"


@dataclass
class ConversationTurn:
    role: str  # "user" | "agent"
    text: str
    turn_index: int
    created_at: datetime = field(default_factory=utcnow)


@dataclass
class Session:
    session_id: str
    user_id: str
    status: SessionStatus = SessionStatus.ACTIVE
    #: connection_id of the single websocket currently allowed to drive this
    #: session, or None while disconnected.
    active_connection_id: str | None = None
    conversation_history: list[ConversationTurn] = field(default_factory=list)
    created_at: datetime = field(default_factory=utcnow)
    last_active_at: datetime = field(default_factory=utcnow)
    disconnected_at: datetime | None = None

    @property
    def turn_number(self) -> int:
        return len(self.conversation_history)

    def append_turn(self, role: str, text: str) -> ConversationTurn:
        turn = ConversationTurn(role=role, text=text, turn_index=self.turn_number)
        self.conversation_history.append(turn)
        self.last_active_at = utcnow()
        return turn
