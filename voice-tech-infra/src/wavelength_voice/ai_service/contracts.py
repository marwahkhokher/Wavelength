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
ModeCategory = Literal["professional", "personal"]
DifficultyLevel = Literal["easy", "medium", "hard"]
PrimaryEmotion = Literal[
    "confident", "anxious", "hesitant", "monotone", "energetic", "empathetic", "neutral", "frustrated"
]


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


# =====================================================================
# AI Team Extended Contracts (v2.1)
# =====================================================================

class PersonaProfile(BaseModel):
    name: str
    role_description: str
    communication_style: str
    attitude: str = "neutral"
    tone_traits: list[str] = Field(default_factory=list)


class ScenarioConfig(BaseModel):
    title: str
    description: str
    setting: str


class BaselineMetrics(BaseModel):
    clarity: float = 6.5
    empathy: float = 7.0
    filler_words: float = 8.0
    confidence: float = 6.0
    structure: float = 5.5
    relevance: float = 7.0


class SessionContext(BaseModel):
    session_id: str
    user_id: str
    mode: ModeCategory
    difficulty: DifficultyLevel = "medium"
    duration_seconds: int = 300
    scenario: ScenarioConfig
    persona: PersonaProfile
    baseline_metrics: BaselineMetrics = Field(default_factory=BaselineMetrics)


class FillerWordOccurrence(BaseModel):
    word: str
    start_time: float
    end_time: float


class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float


class STTResult(BaseModel):
    transcript: str
    is_final: bool = True
    total_words: int
    filler_word_count: int
    filler_words: list[FillerWordOccurrence] = Field(default_factory=list)
    word_timestamps: list[WordTimestamp] = Field(default_factory=list)
    utterance_duration_sec: float
    stt_latency_ms: float = 0.0


class PauseMetrics(BaseModel):
    total_pause_duration_sec: float
    pause_count: int
    max_pause_sec: float


class ToneResult(BaseModel):
    primary_emotion: PrimaryEmotion
    emotion_confidence_scores: dict[str, float] = Field(default_factory=dict)
    pitch_energy_variation: float = Field(ge=0.0, le=1.0, default=0.5)
    hesitation_score: float = Field(ge=0.0, le=1.0, default=0.0)
    pause_metrics: PauseMetrics
    speech_rate_wpm: float
    silence_ratio: float = Field(ge=0.0, le=1.0, default=0.0)


class MetricScores(BaseModel):
    clarity: float = Field(ge=0.0, le=100.0)
    empathy: float = Field(ge=0.0, le=100.0)
    filler_words_score: float = Field(ge=0.0, le=100.0)
    structure: float = Field(ge=0.0, le=100.0)
    relevance: float = Field(ge=0.0, le=100.0)
    confidence: float = Field(ge=0.0, le=100.0)
    overall_turn_score: float = Field(ge=0.0, le=100.0)


class PerTurnEvaluation(BaseModel):
    turn_index: int
    scores: MetricScores
    strengths: list[str] = Field(default_factory=list)
    areas_for_improvement: list[str] = Field(default_factory=list)
    coach_tip: str


class FinalSessionEvaluation(BaseModel):
    session_id: str
    total_turns: int
    average_scores: MetricScores
    score_deltas_from_baseline: dict[str, float]
    coach_mode_summary: list[str]


class TurnRecord(BaseModel):
    turn_index: int
    user_transcript: str
    user_tone: str
    ai_response: str
    eval_scores: MetricScores | None = None


class PromptLLMInput(BaseModel):
    session_context: SessionContext
    current_stt: STTResult
    current_tone: ToneResult
    conversation_history: list[TurnRecord] = Field(default_factory=list)


class PromptLLMOutput(BaseModel):
    reply_text: str
    end_session: bool = False
    persona_state_update: str | None = None
