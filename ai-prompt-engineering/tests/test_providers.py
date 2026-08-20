"""Unit tests for Provider adapters (Mock and Real) (Taha's ownership)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-tech-infra" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prompt_orchestration.providers.mock_providers import (
    MockAnalysisProvider,
    MockSessionContextProvider,
    MockToneProvider,
    MockTranscriptProvider,
)
from prompt_orchestration.providers.real_providers import (
    QwenAnalysisProvider,
    RealSessionContextProvider,
    RealToneProvider,
    WhisperTranscriptProvider,
)


def test_mock_providers():
    async def _run():
        ctx_prov = MockSessionContextProvider()
        ctx = ctx_prov.build_context("s_mock", "u_mock", "professional")
        assert ctx.session_id == "s_mock"
        assert ctx.persona.name == "David Miller"

        audio_bytes = b"\x00" * 3200
        stt_prov = MockTranscriptProvider()
        stt_res = await stt_prov.transcribe(audio_bytes)
        assert stt_res.transcript is not None
        assert stt_res.total_words > 0

        tone_prov = MockToneProvider()
        tone_res = await tone_prov.analyze_tone(audio_bytes, stt_res.total_words, stt_res.utterance_duration_sec)
        assert tone_res.primary_emotion in ["neutral", "confident", "hesitant", "anxious"]
        assert tone_res.speech_rate_wpm >= 0

        analysis_prov = MockAnalysisProvider()
        eval_res = await analysis_prov.evaluate_turn(1, stt_res, tone_res)
        assert eval_res.scores.overall_turn_score >= 0
        assert len(eval_res.strengths) > 0

    asyncio.run(_run())


def test_real_provider_adapters():
    async def _run():
        ctx_prov = RealSessionContextProvider()
        ctx = ctx_prov.build_context(
            session_id="s_real",
            user_id="u_real",
            mode="professional",
            scenario_title="Interview",
            scenario_description="Engineering manager",
            persona_name="Alex",
            persona_role="Director",
        )
        assert ctx.session_id == "s_real"
        assert ctx.persona.name == "Alex"

        audio_bytes = b"\x00" * 16000 * 2
        stt_prov = WhisperTranscriptProvider()
        stt_res = await stt_prov.transcribe(audio_bytes)
        assert stt_res.total_words > 0
        assert "increased team velocity" in stt_res.transcript

        tone_prov = RealToneProvider()
        tone_res = await tone_prov.analyze_tone(audio_bytes, stt_res.total_words, stt_res.utterance_duration_sec)
        assert tone_res.primary_emotion == "hesitant"

        qwen_prov = QwenAnalysisProvider()
        eval_res = await qwen_prov.evaluate_turn(1, stt_res, tone_res)
        assert eval_res.scores.clarity == 78.0
        assert "STAR method" in eval_res.coach_tip or "Pause" in eval_res.coach_tip

    asyncio.run(_run())


if __name__ == "__main__":
    test_mock_providers()
    test_real_provider_adapters()
    print("All provider tests passed!")
