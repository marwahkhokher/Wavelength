from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest


class FakeClock:
    """Deterministic, manually-advanced clock for testing time-based logic."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()
