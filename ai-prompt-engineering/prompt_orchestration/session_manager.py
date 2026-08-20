"""Session Manager — in-memory session state (Taha's ownership).

Manages ConversationState lifecycle: creation, retrieval, update, cleanup.
Sessions are stored in-memory (suitable for single-process MVP).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .models import ConversationState, SessionState

logger = logging.getLogger(__name__)


class SessionManager:
    """In-memory store for active conversation sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, ConversationState] = {}

    def create_session(self, session_id: str, user_id: str) -> ConversationState:
        """Create a new conversation session."""
        if session_id in self._sessions:
            logger.warning("Session %s already exists, returning existing", session_id)
            return self._sessions[session_id]

        state = ConversationState(
            session_id=session_id,
            user_id=user_id,
            state=SessionState.ACTIVE,
        )
        self._sessions[session_id] = state
        logger.info("Created session %s for user %s", session_id, user_id)
        return state

    def get_session(self, session_id: str) -> ConversationState | None:
        """Retrieve an active session by ID."""
        return self._sessions.get(session_id)

    def get_or_create_session(self, session_id: str, user_id: str) -> ConversationState:
        """Get existing session or create a new one."""
        existing = self.get_session(session_id)
        if existing is not None:
            return existing
        return self.create_session(session_id, user_id)

    def update_session(self, session_id: str, updates: dict[str, Any]) -> ConversationState | None:
        """Apply partial updates to a session."""
        session = self.get_session(session_id)
        if session is None:
            return None
        for key, value in updates.items():
            if hasattr(session, key):
                setattr(session, key, value)
        session.last_activity = time.time()
        return session

    def end_session(self, session_id: str) -> ConversationState | None:
        """Mark a session as completed."""
        session = self.get_session(session_id)
        if session is None:
            return None
        session.state = SessionState.COMPLETED
        logger.info("Ended session %s (turns: %d)", session_id, session.current_turn)
        return session

    def remove_session(self, session_id: str) -> bool:
        """Remove a session from memory."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(self) -> list[str]:
        """Return all active session IDs."""
        return list(self._sessions.keys())

    def cleanup_stale(self, max_age_seconds: float = 3600) -> int:
        """Remove sessions inactive for more than max_age_seconds."""
        now = time.time()
        stale = [
            sid for sid, s in self._sessions.items()
            if (now - s.last_activity) > max_age_seconds
        ]
        for sid in stale:
            del self._sessions[sid]
        if stale:
            logger.info("Cleaned up %d stale sessions", len(stale))
        return len(stale)
