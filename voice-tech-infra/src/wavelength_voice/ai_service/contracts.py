"""Request/response contract between Voice/Tech Infra and the AI service.

This is the boundary the AI/Prompt Engineering team's real service must
implement. Voice/Tech Infra only ever talks to this contract - never to
prompt internals - so either side can change independently as long as
these shapes hold.

Wire format: JSON over HTTP POST {base_url}/v1/turn (see
``HTTPAIServiceClient``). The same models are used in-process by
``MockAIServiceClient`` so unit tests exercise the exact contract.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["user", "agent"]


class PersonaConfig(BaseModel):
    """Describes the AI persona the user is practicing a conversation with."""

    persona_id: str
    name: str
    scenario_prompt: str
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    traits: list[str] = Field(default_factory=list)


class ConversationTurnDTO(BaseModel):
    """One turn of conversation history, as sent to the AI service."""

    role: Role
    text: str
    turn_index: int


class AITurnRequest(BaseModel):
    """Sent to the AI service once a user's utterance is finalized by STT."""

    session_id: str
    user_id: str
    persona: PersonaConfig
    transcript: str
    conversation_history: list[ConversationTurnDTO] = Field(default_factory=list)
    turn_number: int = Field(ge=0)


class AITurnResponse(BaseModel):
    """Returned by the AI service; drives what the agent says next via TTS."""

    reply_text: str
    persona_state: dict[str, str] = Field(default_factory=dict)
    end_session: bool = False
    feedback_hint: str | None = None
    latency_ms: int | None = None
