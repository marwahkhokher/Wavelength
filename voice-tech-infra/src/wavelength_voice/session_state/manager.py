"""In-memory session state manager.

Handles the two resilience behaviours a voice UI needs from the server side:

* **Reconnect** - a dropped websocket (flaky wifi, phone lock, tab refresh)
  doesn't lose the conversation. The session survives for
  ``reconnect_grace_seconds`` after a disconnect; a reconnect within that
  window resumes the same session with its full history intact.
* **Duplicate tab** - if the same session is opened in a second tab/device,
  the new connection takes over ("last tab wins") and the caller is told
  which old connection to evict, rather than silently running two
  concurrent conversations against one session.

This module is pure state + policy - it knows nothing about websockets or
FastAPI, which is what makes it unit-testable without a network stack. The
websocket router (``wavelength_voice.ws.router``) is the only caller and is
responsible for actually closing evicted connections.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from wavelength_voice.session_state.models import (
    ConversationTurn,
    Session,
    SessionStatus,
    utcnow,
)

#: Reasons a connect() call can report back to the caller.
ConnectReason = str


@dataclass
class ConnectResult:
    accepted: bool
    reason: ConnectReason
    session: Session | None
    connection_id: str
    #: Set when this connect took over from a still-registered connection,
    #: which the caller must forcibly close.
    evicted_connection_id: str | None = None


class SessionManager:
    """Owns all live/recently-live sessions for this server process."""

    def __init__(
        self,
        reconnect_grace_seconds: float = 60.0,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._grace_seconds = reconnect_grace_seconds
        self._now = clock
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def connect(
        self, session_id: str, connection_id: str, user_id: str
    ) -> ConnectResult:
        """Attach a websocket connection to a session, creating it if new.

        A second connection to a session that already has an active
        connection is treated as a duplicate tab: it wins, and the previous
        connection_id is returned as ``evicted_connection_id`` so the caller
        can close it out from under the old tab.
        """
        async with self._lock:
            session = self._sessions.get(session_id)

            if session is None:
                session = Session(session_id=session_id, user_id=user_id)
                self._sessions[session_id] = session
                session.active_connection_id = connection_id
                session.last_active_at = self._now()
                return ConnectResult(
                    accepted=True,
                    reason="new_session",
                    session=session,
                    connection_id=connection_id,
                )

            if session.status in (SessionStatus.EXPIRED, SessionStatus.ENDED):
                return ConnectResult(
                    accepted=False,
                    reason=f"session_{session.status.value}",
                    session=session,
                    connection_id=connection_id,
                )

            evicted_connection_id = None
            reason = "reconnected"
            if (
                session.status == SessionStatus.ACTIVE
                and session.active_connection_id is not None
                and session.active_connection_id != connection_id
            ):
                evicted_connection_id = session.active_connection_id
                reason = "duplicate_tab_takeover"

            session.active_connection_id = connection_id
            session.status = SessionStatus.ACTIVE
            session.disconnected_at = None
            session.last_active_at = self._now()

            return ConnectResult(
                accepted=True,
                reason=reason,
                session=session,
                connection_id=connection_id,
                evicted_connection_id=evicted_connection_id,
            )

    async def disconnect(self, session_id: str, connection_id: str) -> bool:
        """Detach a connection, starting the reconnect grace window.

        Returns False (a no-op) if ``connection_id`` was not the session's
        current active connection - this happens when an evicted duplicate
        tab's connection finally tears down after having already lost a
        takeover race, and it must not be allowed to clobber the new
        connection's state.
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.active_connection_id != connection_id:
                return False

            session.active_connection_id = None
            session.status = SessionStatus.DISCONNECTED_GRACE
            session.disconnected_at = self._now()
            return True

    async def end_session(self, session_id: str) -> None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            session.status = SessionStatus.ENDED
            session.active_connection_id = None

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def append_turn(self, session_id: str, role: str, text: str) -> ConversationTurn:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Unknown session_id: {session_id}")
        return session.append_turn(role, text)

    def sweep_expired(self) -> list[str]:
        """Expire sessions whose reconnect grace window has elapsed.

        Safe to call frequently; cheap and side-effect-free for sessions
        that aren't in the grace window. Returns the ids that expired in
        this sweep.
        """
        now = self._now()
        expired_ids: list[str] = []
        for session in self._sessions.values():
            if (
                session.status == SessionStatus.DISCONNECTED_GRACE
                and session.disconnected_at is not None
                and (now - session.disconnected_at).total_seconds()
                >= self._grace_seconds
            ):
                session.status = SessionStatus.EXPIRED
                expired_ids.append(session.session_id)
        return expired_ids

    async def run_expiry_sweeper(self, interval_seconds: float) -> None:
        """Background loop: call from a FastAPI startup task. Never returns."""
        while True:
            await asyncio.sleep(interval_seconds)
            async with self._lock:
                self.sweep_expired()
