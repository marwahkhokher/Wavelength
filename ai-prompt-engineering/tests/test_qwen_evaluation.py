"""Tests for the deterministic Qwen evaluation integration."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-tech-infra" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwen_evaluation.deep_evaluator import QwenDeepEvaluator
from qwen_evaluation.coaching_engine import CoachingEngine
from qwen_evaluation.qwen_client import LiveQwenEvaluationClient
from qwen_evaluation.scorecard_generator import generate_session_scorecard
from session_context.builder import build_session_context
from wavelength_voice.ai_service.contracts import (
    BaselineMetrics,
    FillerWordOccurrence,
    PauseMetrics,
    STTResult,
    ToneResult,
)


class FakeLiveClient:
    def __init__(self, result):  # noqa: ANN001
        self.result = result
        self.payload = None

    async def evaluate(self, payload):  # noqa: ANN001
        self.payload = payload
        return self.result


class FakeQwenTransport:
    def __init__(self, content: str) -> None:
        self.content = content
        self.kwargs = None
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs):  # noqa: ANN003
        self.kwargs = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


def make_tone(*, emotion: str = "confident", hesitation: float = 0.1) -> ToneResult:
    return ToneResult(
        primary_emotion=emotion,  # type: ignore[arg-type]
        emotion_confidence_scores={emotion: 0.9},
        hesitation_score=hesitation,
        pitch_energy_variation=0.5,
        pause_metrics=PauseMetrics(total_pause_duration_sec=0.2, pause_count=1, max_pause_sec=0.2),
        speech_rate_wpm=140.0,
        silence_ratio=0.02,
    )


def make_context():
    return build_session_context(
        session_id="session-1",
        user_id="user-1",
        mode="professional",
        scenario_title="Salary negotiation",
        scenario_description="Discuss a performance-based salary increase with a manager.",
        persona_name="Alex",
        persona_role="Manager",
    )


def test_evaluator_uses_stt_tone_and_scenario_context() -> None:
    stt = STTResult(
        transcript="First, I improved delivery by automating the release process, and the result was fewer delays.",
        total_words=15,
        filler_word_count=0,
        utterance_duration_sec=6.5,
    )

    evaluation = asyncio.run(
        QwenDeepEvaluator().evaluate_turn(1, stt, make_tone(), make_context())
    )

    assert evaluation.scores.confidence > 80
    assert evaluation.scores.fluency > 90
    assert evaluation.scores.filler_words_score == 100
    assert evaluation.main_point_detected.startswith("First, I improved")
    assert evaluation.suggested_conversation_followup_direction


def test_evaluator_identifies_fillers_and_hesitation() -> None:
    stt = STTResult(
        transcript="Um, like, I think it was good.",
        total_words=7,
        filler_word_count=2,
        filler_words=[
            FillerWordOccurrence(word="um", start_time=0.0, end_time=0.2),
            FillerWordOccurrence(word="like", start_time=0.3, end_time=0.5),
        ],
        utterance_duration_sec=5.0,
    )

    evaluation = asyncio.run(
        QwenDeepEvaluator().evaluate_turn(1, stt, make_tone(emotion="hesitant", hesitation=0.8))
    )

    assert evaluation.scores.filler_words_score < 70
    assert evaluation.scores.confidence < 50
    assert evaluation.areas_for_improvement
    assert "filler" in evaluation.coach_tip.lower()


def test_scorecard_includes_fluency_and_filler_baseline_deltas() -> None:
    stt = STTResult(
        transcript="I delivered a measurable result.",
        total_words=5,
        filler_word_count=0,
        utterance_duration_sec=2.0,
    )
    evaluation = asyncio.run(QwenDeepEvaluator().evaluate_turn(1, stt, make_tone()))

    scorecard = generate_session_scorecard(
        "session-1", [evaluation], BaselineMetrics(fluency=6.0, filler_words=6.0)
    )

    assert scorecard.average_scores.fluency == evaluation.scores.fluency
    assert "fluency_delta" in scorecard.score_deltas_from_baseline
    assert "filler_words_delta" in scorecard.score_deltas_from_baseline


def test_coaching_report_uses_the_original_answer_without_inventing_results() -> None:
    stt = STTResult(
        transcript="Um, like, I improved the release process.",
        total_words=7,
        filler_word_count=2,
        utterance_duration_sec=4.0,
    )
    evaluation = asyncio.run(
        QwenDeepEvaluator().evaluate_turn(1, stt, make_tone(emotion="hesitant", hesitation=0.8))
    )

    report = CoachingEngine().generate_coaching_report("session-1", [stt], [evaluation])

    refinement = report["answer_refinements"][0]
    assert "improved the release process" in refinement["refined_better_answer"].lower()
    assert "25%" not in refinement["refined_better_answer"]
    assert report["improvement_action_plan"]


def test_evaluator_uses_live_provider_when_supplied() -> None:
    stt = STTResult(
        transcript="I delivered the agreed outcome.",
        total_words=5,
        filler_word_count=0,
        utterance_duration_sec=2.0,
    )
    expected = asyncio.run(QwenDeepEvaluator().evaluate_turn(1, stt, make_tone()))
    provider = FakeLiveClient(expected)

    actual = asyncio.run(QwenDeepEvaluator(live_client=provider).evaluate_turn(3, stt, make_tone()))

    assert actual == expected
    assert provider.payload["turn_index"] == 3
    assert provider.payload["stt_result"]["transcript"] == stt.transcript


def test_live_client_parses_fenced_json() -> None:
    content = "```json\n{\"turn_index\": 1, \"scores\": {\"clarity\": 80, \"empathy\": 70, \"filler_words_score\": 90, \"structure\": 75, \"relevance\": 85, \"confidence\": 80, \"fluency\": 88, \"overall_turn_score\": 81}, \"strengths\": [], \"areas_for_improvement\": [], \"coach_tip\": \"Keep going.\"}\n```"

    parsed = LiveQwenEvaluationClient._parse_json(content)

    assert parsed["turn_index"] == 1


def test_live_client_sends_structured_payload_to_qwen_transport() -> None:
    content = '{"turn_index": 1, "scores": {"clarity": 80, "empathy": 70, "filler_words_score": 90, "structure": 75, "relevance": 85, "confidence": 80, "fluency": 88, "overall_turn_score": 81}, "strengths": [], "areas_for_improvement": [], "coach_tip": "Keep going."}'
    transport = FakeQwenTransport(content)
    client = LiveQwenEvaluationClient(client=transport, model="qwen-test")

    evaluation = asyncio.run(client.evaluate({"turn_index": 1, "transcript": "Hello"}))

    assert evaluation.scores.overall_turn_score == 81
    assert transport.kwargs["model"] == "qwen-test"
    assert transport.kwargs["messages"][1]["content"] == '{"turn_index": 1, "transcript": "Hello"}'
