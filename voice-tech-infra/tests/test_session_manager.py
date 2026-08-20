"""Tests for session resilience: reconnect-with-grace and duplicate-tab handling."""

from __future__ import annotations

import asyncio

import pytest

from wavelength_voice.session_state.manager import SessionManager
from wavelength_voice.session_state.models import SessionStatus


@pytest.fixture
def manager(fake_clock) -> SessionManager:
    return SessionManager(reconnect_grace_seconds=60.0, clock=fake_clock)


async def test_first_connect_creates_a_new_session(manager: SessionManager) -> None:
    result = await manager.connect("s1", "conn-a", user_id="u1")

    assert result.accepted
    assert result.reason == "new_session"
    assert result.evicted_connection_id is None
    session = manager.get_session("s1")
    assert session is not None
    assert session.status == SessionStatus.ACTIVE
    assert session.active_connection_id == "conn-a"
    assert session.user_id == "u1"


async def test_disconnect_starts_grace_window_and_preserves_history(
    manager: SessionManager,
) -> None:
    await manager.connect("s1", "conn-a", user_id="u1")
    manager.append_turn("s1", "user", "hello")

    applied = await manager.disconnect("s1", "conn-a")

    assert applied is True
    session = manager.get_session("s1")
    assert session.status == SessionStatus.DISCONNECTED_GRACE
    assert session.active_connection_id is None
    assert [t.text for t in session.conversation_history] == ["hello"]


async def test_reconnect_within_grace_window_resumes_same_session(
    manager: SessionManager, fake_clock
) -> None:
    await manager.connect("s1", "conn-a", user_id="u1")
    manager.append_turn("s1", "user", "hello")
    await manager.disconnect("s1", "conn-a")

    fake_clock.advance(30)  # well within the 60s grace window
    result = await manager.connect("s1", "conn-b", user_id="u1")

    assert result.accepted
    assert result.reason == "reconnected"
    session = result.session
    assert session.status == SessionStatus.ACTIVE
    assert session.active_connection_id == "conn-b"
    assert [t.text for t in session.conversation_history] == ["hello"]


async def test_session_expires_after_grace_window_elapses(
    manager: SessionManager, fake_clock
) -> None:
    await manager.connect("s1", "conn-a", user_id="u1")
    await manager.disconnect("s1", "conn-a")

    fake_clock.advance(61)  # just past the 60s grace window
    expired_ids = manager.sweep_expired()

    assert expired_ids == ["s1"]
    assert manager.get_session("s1").status == SessionStatus.EXPIRED


async def test_sweep_expired_leaves_sessions_within_grace_window_alone(
    manager: SessionManager, fake_clock
) -> None:
    await manager.connect("s1", "conn-a", user_id="u1")
    await manager.disconnect("s1", "conn-a")

    fake_clock.advance(59)
    expired_ids = manager.sweep_expired()

    assert expired_ids == []
    assert manager.get_session("s1").status == SessionStatus.DISCONNECTED_GRACE


async def test_connect_to_expired_session_is_rejected(
    manager: SessionManager, fake_clock
) -> None:
    await manager.connect("s1", "conn-a", user_id="u1")
    await manager.disconnect("s1", "conn-a")
    fake_clock.advance(61)
    manager.sweep_expired()

    result = await manager.connect("s1", "conn-b", user_id="u1")

    assert result.accepted is False
    assert result.reason == "session_expired"


async def test_second_connection_is_a_duplicate_tab_takeover(
    manager: SessionManager,
) -> None:
    first = await manager.connect("s1", "conn-a", user_id="u1")
    second = await manager.connect("s1", "conn-b", user_id="u1")

    assert first.accepted and second.accepted
    assert second.reason == "duplicate_tab_takeover"
    assert second.evicted_connection_id == "conn-a"
    session = manager.get_session("s1")
    assert session.active_connection_id == "conn-b"
    assert session.status == SessionStatus.ACTIVE


async def test_evicted_connections_disconnect_is_a_noop(manager: SessionManager) -> None:
    await manager.connect("s1", "conn-a", user_id="u1")
    await manager.connect("s1", "conn-b", user_id="u1")  # evicts conn-a

    # The old tab's websocket handler finally notices it was closed and
    # calls disconnect() for conn-a - this must NOT tear down conn-b's session.
    applied = await manager.disconnect("s1", "conn-a")

    assert applied is False
    session = manager.get_session("s1")
    assert session.status == SessionStatus.ACTIVE
    assert session.active_connection_id == "conn-b"


async def test_disconnect_on_unknown_session_returns_false(manager: SessionManager) -> None:
    assert await manager.disconnect("does-not-exist", "conn-a") is False


async def test_concurrent_connects_to_the_same_session_only_one_wins_as_active(
    manager: SessionManager,
) -> None:
    await manager.connect("s1", "conn-a", user_id="u1")

    # Two "duplicate tabs" racing to connect at once.
    results = await asyncio.gather(
        manager.connect("s1", "conn-b", user_id="u1"),
        manager.connect("s1", "conn-c", user_id="u1"),
    )

    session = manager.get_session("s1")
    # Exactly one of conn-b/conn-c ends up active; the lock serializes the race.
    assert session.active_connection_id in {"conn-b", "conn-c"}
    winner, loser = (
        ("conn-b", "conn-c")
        if session.active_connection_id == "conn-b"
        else ("conn-c", "conn-b")
    )
    evicted = {r.evicted_connection_id for r in results}
    assert loser in evicted or "conn-a" in evicted


async def test_end_session_marks_ended_and_clears_active_connection(
    manager: SessionManager,
) -> None:
    await manager.connect("s1", "conn-a", user_id="u1")

    await manager.end_session("s1")

    session = manager.get_session("s1")
    assert session.status == SessionStatus.ENDED
    assert session.active_connection_id is None


async def test_reconnect_after_end_session_is_rejected(manager: SessionManager) -> None:
    await manager.connect("s1", "conn-a", user_id="u1")
    await manager.end_session("s1")

    result = await manager.connect("s1", "conn-b", user_id="u1")

    assert result.accepted is False
    assert result.reason == "session_ended"


async def test_append_turn_on_unknown_session_raises(manager: SessionManager) -> None:
    with pytest.raises(KeyError):
        manager.append_turn("does-not-exist", "user", "hi")
