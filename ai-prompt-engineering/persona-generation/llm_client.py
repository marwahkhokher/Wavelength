"""
LLM Client — thin abstraction over the provider (architecture doc Part 16:
one provider, behind one interface, so swapping it later is a config
change, not a rewrite). Every module in this pipeline talks to `LLMClient`,
never to a provider SDK directly.

SECURITY: GeminiLLMClient reads its key from the GEMINI_API_KEY
environment variable ONLY. Never hardcode a key here, in a test, or
anywhere else in source - see .env.example for how callers should supply
it locally.
"""

from __future__ import annotations

import abc
import os


class LLMClient(abc.ABC):
    @abc.abstractmethod
    async def complete(self, messages: list[dict[str, str]]) -> str:
        """Return the raw text response for a chat-style message list."""
        raise NotImplementedError


class MockLLMClient(LLMClient):
    """Returns scripted responses in order.

    Used by every test in this module set, and for local development
    without a live API key - the entire pipeline (situation extraction
    through runtime prompt assembly) is exercised end to end in
    test_pipeline.py using this instead of a real provider call.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.calls: list[list[dict[str, str]]] = []

    async def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        try:
            return next(self._responses)
        except StopIteration as exc:
            raise AssertionError(
                f"MockLLMClient ran out of scripted responses after {len(self.calls)} call(s)"
            ) from exc


class GeminiLLMClient(LLMClient):
    """
    Real implementation, written against the documented `google-genai` SDK
    surface. Not exercised against a live API in this repo - no verified
    key is available in this environment (see the security note in the
    architecture doc: a key was pasted into chat and should be treated as
    burned, not used). Treat this the same way as voice-tech-infra's
    DeepgramSTTStream/ElevenLabsTTSStream: written to spec, unverified
    against live traffic until someone runs it with a real key.

    Requires the `google-genai` package, which is intentionally NOT a hard
    dependency of this module (imported lazily below) so the rest of the
    pipeline works without it installed.
    """

    def __init__(self, model: str = "gemini-3.6-flash", api_key: str | None = None) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Put it in a gitignored .env file and load it "
                "before constructing this client - never hardcode a key in source."
            )

    async def complete(self, messages: list[dict[str, str]]) -> str:
        from google import genai  # lazy import - optional dependency until this is wired up

        client = genai.Client(api_key=self._api_key)
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        user = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
        # client.aio, not client.models - the latter is synchronous and would
        # silently block the event loop from inside this async method.
        response = await client.aio.models.generate_content(
            model=self._model,
            contents=user,
            config={"system_instruction": system} if system else None,
        )
        return response.text
