"""AI-service client: the abstract contract, a mock, and a real HTTP client.

Voice/Tech Infra depends only on ``AIServiceClient``. ``MockAIServiceClient``
lets this repo's voice pipeline, session, and turn-taking logic be built and
tested before the AI/Prompt Engineering team's real service exists.
``HTTPAIServiceClient`` is the drop-in replacement once it does - swap it in
via ``config.use_mock_ai_service`` with no other code changes required.
"""

from __future__ import annotations

import abc
import asyncio
import itertools

import httpx

from wavelength_voice.ai_service.contracts import AITurnRequest, AITurnResponse


class AIServiceClient(abc.ABC):
    """Contract every AI-service client implementation must satisfy."""

    @abc.abstractmethod
    async def get_response(self, request: AITurnRequest) -> AITurnResponse:
        """Return the agent's next turn for a finalized user utterance."""
        raise NotImplementedError


class AIServiceError(RuntimeError):
    """Raised when the AI service is unreachable or returns an invalid response."""


class MockAIServiceClient(AIServiceClient):
    """Deterministic in-process stand-in for the real AI service.

    Cycles through canned replies (or a caller-supplied script) so callers
    can exercise multi-turn flows without a network dependency. ``latency_seconds``
    simulates real-world response time, e.g. to test barge-in against a
    "still thinking" agent.
    """

    _DEFAULT_REPLIES = [
        "That's a fair point - tell me more about what led you there.",
        "I hear you, but I'm not sure that fully addresses my concern.",
        "Okay, let's take a step back. What outcome are you actually hoping for?",
    ]

    def __init__(
        self,
        replies: list[str] | None = None,
        latency_seconds: float = 0.0,
        end_after_turns: int | None = None,
    ) -> None:
        self._replies = itertools.cycle(replies or self._DEFAULT_REPLIES)
        self._latency_seconds = latency_seconds
        self._end_after_turns = end_after_turns
        self.calls: list[AITurnRequest] = []

    async def get_response(self, request: AITurnRequest) -> AITurnResponse:
        self.calls.append(request)
        if self._latency_seconds:
            await asyncio.sleep(self._latency_seconds)

        end_session = (
            self._end_after_turns is not None
            and request.turn_number >= self._end_after_turns
        )
        return AITurnResponse(
            reply_text=next(self._replies),
            persona_state={"mood": "engaged"},
            end_session=end_session,
            feedback_hint=None,
            latency_ms=int(self._latency_seconds * 1000),
        )


class HTTPAIServiceClient(AIServiceClient):
    """Real client - POSTs to the AI service's ``/v1/turn`` endpoint."""

    def __init__(self, base_url: str, timeout_seconds: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    async def get_response(self, request: AITurnRequest) -> AITurnResponse:
        url = f"{self._base_url}/v1/turn"
        try:
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                resp = await client.post(url, json=request.model_dump(mode="json"))
                resp.raise_for_status()
                return AITurnResponse.model_validate(resp.json())
        except httpx.HTTPError as exc:
            raise AIServiceError(f"AI service request to {url} failed: {exc}") from exc
        except Exception as exc:  # pydantic ValidationError, JSON decode error, etc.
            raise AIServiceError(
                f"AI service at {url} returned an invalid response: {exc}"
            ) from exc
