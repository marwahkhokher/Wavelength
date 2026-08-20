"""Tests for the AI-service request/response contract and its mock/HTTP clients."""

from __future__ import annotations

import httpx
import pytest
from pydantic import ValidationError

from wavelength_voice.ai_service.client import (
    AIServiceError,
    HTTPAIServiceClient,
    MockAIServiceClient,
)
from wavelength_voice.ai_service.contracts import (
    AITurnRequest,
    AITurnResponse,
    ConversationTurnDTO,
    PersonaConfig,
)

PERSONA = PersonaConfig(
    persona_id="p1", name="Alex", scenario_prompt="A skeptical stakeholder."
)


def make_request(turn_number: int = 0, history: list[ConversationTurnDTO] | None = None) -> AITurnRequest:
    return AITurnRequest(
        session_id="s1",
        user_id="u1",
        persona=PERSONA,
        transcript="Here's my update on the project.",
        conversation_history=history or [],
        turn_number=turn_number,
    )


def test_request_requires_all_fields() -> None:
    with pytest.raises(ValidationError):
        AITurnRequest(session_id="s1")  # type: ignore[call-arg]


def test_request_rejects_negative_turn_number() -> None:
    with pytest.raises(ValidationError):
        make_request(turn_number=-1)


async def test_mock_client_returns_a_contract_valid_response() -> None:
    client = MockAIServiceClient()
    request = make_request()

    response = await client.get_response(request)

    assert isinstance(response, AITurnResponse)
    assert response.reply_text
    assert response.end_session is False
    assert client.calls == [request]


async def test_mock_client_cycles_through_scripted_replies() -> None:
    client = MockAIServiceClient(replies=["first", "second"])

    r1 = await client.get_response(make_request(turn_number=0))
    r2 = await client.get_response(make_request(turn_number=1))
    r3 = await client.get_response(make_request(turn_number=2))

    assert [r1.reply_text, r2.reply_text, r3.reply_text] == ["first", "second", "first"]


async def test_mock_client_signals_end_session_after_configured_turns() -> None:
    client = MockAIServiceClient(end_after_turns=2)

    r1 = await client.get_response(make_request(turn_number=1))
    r2 = await client.get_response(make_request(turn_number=2))

    assert r1.end_session is False
    assert r2.end_session is True


async def test_mock_client_simulates_latency() -> None:
    client = MockAIServiceClient(latency_seconds=0.01)

    response = await client.get_response(make_request())

    assert response.latency_ms == 10


async def test_http_client_posts_the_contract_and_parses_a_valid_response() -> None:
    captured: dict = {}

    client = HTTPAIServiceClient(base_url="http://ai.internal")
    request = make_request()

    async def fake_post(self, url, json):  # noqa: ANN001
        captured["url"] = url
        captured["json"] = json
        return httpx.Response(
            200,
            json={
                "reply_text": "Interesting, walk me through the tradeoffs.",
                "persona_state": {"mood": "curious"},
                "end_session": False,
            },
            request=httpx.Request("POST", url),
        )

    # Patch httpx.AsyncClient.post for this test rather than hitting the network.
    original_post = httpx.AsyncClient.post
    httpx.AsyncClient.post = fake_post  # type: ignore[method-assign]
    try:
        response = await client.get_response(request)
    finally:
        httpx.AsyncClient.post = original_post  # type: ignore[method-assign]

    assert captured["url"] == "http://ai.internal/v1/turn"
    assert captured["json"]["session_id"] == "s1"
    assert captured["json"]["persona"]["persona_id"] == "p1"
    assert response.reply_text == "Interesting, walk me through the tradeoffs."


async def test_http_client_raises_ai_service_error_on_invalid_response_body() -> None:
    client = HTTPAIServiceClient(base_url="http://ai.internal")

    async def fake_post(self, url, json):  # noqa: ANN001
        return httpx.Response(
            200,
            json={"reply_text": None},  # missing/invalid shape
            request=httpx.Request("POST", url),
        )

    original_post = httpx.AsyncClient.post
    httpx.AsyncClient.post = fake_post  # type: ignore[method-assign]
    try:
        with pytest.raises(AIServiceError):
            await client.get_response(make_request())
    finally:
        httpx.AsyncClient.post = original_post  # type: ignore[method-assign]


async def test_http_client_raises_ai_service_error_on_http_failure() -> None:
    client = HTTPAIServiceClient(base_url="http://ai.internal")

    async def fake_post(self, url, json):  # noqa: ANN001
        return httpx.Response(500, json={"error": "boom"}, request=httpx.Request("POST", url))

    original_post = httpx.AsyncClient.post
    httpx.AsyncClient.post = fake_post  # type: ignore[method-assign]
    try:
        with pytest.raises(AIServiceError):
            await client.get_response(make_request())
    finally:
        httpx.AsyncClient.post = original_post  # type: ignore[method-assign]
