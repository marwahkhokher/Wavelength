"""Integration test suite for AI team modules on branch team/ai."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add project roots to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-tech-infra" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from session_context.builder import build_session_context
from session_context.tone_rules import get_mode_tone_prompt
from stt_filler.filler_detector import FillerDetector
from stt_filler.stt_engine import WhisperSTTEngine
from tone_analysis.acoustic_metrics import calculate_acoustic_metrics
from tone_analysis.emotion_classifier import EmotionClassifier
from qwen_evaluation.deep_evaluator import QwenDeepEvaluator
from qwen_evaluation.scorecard_generator import generate_session_scorecard
from qwen_evaluation.coaching_engine import CoachingEngine
from prompt_orchestration.conversation_llm import ConversationLLM
from prompt_orchestration.pipeline_orchestrator import VoicePipelineOrchestrator


def test_session_context_builder():
    ctx = build_session_context(
        session_id="test_1",
        user_id="usr_1",
        mode="professional",
        scenario_title="Conversation",
        scenario_description="Job conversation",
        persona_name="Alex",
        persona_role="Hiring Manager",
    )
    assert ctx.session_id == "test_1"
    assert ctx.mode == "professional"
    assert "workplace norms" in ctx.persona.communication_style


def test_tone_rules():
    prof_rules = get_mode_tone_prompt("professional")
    pers_rules = get_mode_tone_prompt("personal")
    assert "diplomatically" in prof_rules
    assert "Yaar" in pers_rules


def test_filler_detection():
    detector = FillerDetector()
    fillers = detector.detect_fillers("Um, I think like we should start.")
    assert len(fillers) >= 2
    assert fillers[0].word == "um"


def test_acoustic_metrics():
    metrics = calculate_acoustic_metrics(word_count=20, duration_sec=10.0)
    assert metrics["speech_rate_wpm"] == 120.0


def test_full_pipeline_execution():
    async def run_test():
        ctx = build_session_context(
            session_id="test_seq",
            user_id="usr_seq",
            mode="professional",
            scenario_title="Negotiation",
            scenario_description="Salary raise",
            persona_name="David",
            persona_role="VP",
        )
        orchestrator = VoicePipelineOrchestrator(ctx)
        fake_audio = b"\x00" * 16000 * 2  # 1 sec audio
        
        output = await orchestrator.process_user_turn(fake_audio)
        assert output.reply_text is not None
        assert len(orchestrator.conversation_history) == 1

    asyncio.run(run_test())


if __name__ == "__main__":
    test_session_context_builder()
    test_tone_rules()
    test_filler_detection()
    test_acoustic_metrics()
    test_full_pipeline_execution()
    print("All AI team module unit tests passed successfully!")
