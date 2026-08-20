"""Deployable FastAPI boundary for the AI/Prompt Engineering service.

Run locally from the repository root:
``.venv\\Scripts\\python -m uvicorn service:app --app-dir ai-prompt-engineering``
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
VOICE_SRC = ROOT.parent / "voice-tech-infra" / "src"
for path in (ROOT, VOICE_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

load_dotenv(ROOT.parent / ".env")

from prompt_orchestration.conversation_llm import ConversationLLM
from qwen_evaluation.deep_evaluator import QwenDeepEvaluator
from wavelength_voice.ai_service.contracts import (
    AITurnRequest,
    AITurnResponse,
    PauseMetrics,
    PersonaProfile,
    PromptLLMInput,
    ScenarioConfig,
    SessionContext,
    STTResult,
    ToneResult,
)

app = FastAPI(title="Wavelength AI Service", version="1.0.0")
evaluator = QwenDeepEvaluator()
conversation_llm = ConversationLLM()


def build_session_context(request: AITurnRequest) -> SessionContext:
    if request.session_context is not None:
        return request.session_context
    return SessionContext(
        session_id=request.session_id,
        user_id=request.user_id,
        mode="professional",
        difficulty=request.persona.difficulty,
        scenario=ScenarioConfig(
            title="Live conversation",
            description=request.persona.scenario_prompt,
            setting="Voice session",
        ),
        persona=PersonaProfile(
            name=request.persona.name,
            role_description=request.persona.scenario_prompt,
            communication_style="Measured and professional.",
            tone_traits=request.persona.traits,
        ),
    )


def build_stt_result(request: AITurnRequest) -> STTResult:
    if request.stt_result is not None:
        return request.stt_result
    words = request.transcript.split()
    return STTResult(
        transcript=request.transcript,
        total_words=len(words),
        filler_word_count=0,
        utterance_duration_sec=max(1.0, len(words) / 2.3),
    )


def build_tone_result(request: AITurnRequest) -> ToneResult:
    if request.tone_result is not None:
        return request.tone_result
    return ToneResult(
        primary_emotion="neutral",
        pause_metrics=PauseMetrics(
            total_pause_duration_sec=0.0,
            pause_count=0,
            max_pause_sec=0.0,
        ),
        speech_rate_wpm=140.0,
    )


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/turn", response_model=AITurnResponse)
async def process_turn(request: AITurnRequest) -> AITurnResponse:
    """Evaluate one user turn and generate the next in-character response."""
    session_context = build_session_context(request)
    stt_result = build_stt_result(request)
    tone_result = build_tone_result(request)
    evaluation = await evaluator.evaluate_turn(
        turn_index=request.turn_number,
        stt_result=stt_result,
        tone_result=tone_result,
        session_context=session_context,
    )
    output = await conversation_llm.generate_next_turn(
        PromptLLMInput(
            session_context=session_context,
            current_stt=stt_result,
            current_tone=tone_result,
        ),
        evaluation,
    )
    return AITurnResponse(
        reply_text=output.reply_text,
        persona_state={"evaluation_score": str(evaluation.scores.overall_turn_score)},
        end_session=output.end_session,
        feedback_hint=evaluation.coach_tip,
        evaluation=evaluation,
    )
