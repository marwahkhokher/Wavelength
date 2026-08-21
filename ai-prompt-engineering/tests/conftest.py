"""Shared test fixtures for the AI/Prompt Engineering test suite."""

from __future__ import annotations

import pytest

_QWEN_ENV_VARS = ("QWEN_BASE_URL", "QWEN_API_KEY", "QWEN_MODEL", "DASHSCOPE_API_KEY")


@pytest.fixture(autouse=True)
def _isolate_qwen_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests deterministic regardless of a real local .env.

    test_ai_service.py imports service.py, which calls load_dotenv() and
    mutates os.environ for the rest of the pytest process. Without this
    fixture, a real QWEN_BASE_URL in the repo-root .env leaks into every
    other test module, silently turning "deterministic" evaluator tests
    into real (and very slow) live Qwen calls.
    """
    for name in _QWEN_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
