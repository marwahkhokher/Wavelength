"""Integration tests for ConversationOrchestrator and streaming TTS (Taha's ownership)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-tech-infra" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prompt_orchestration.conversation_orchestrator import ConversationOrchestrator
from prompt_orchestration.models import SessionState
from prompt_orchestration.tts_stream import MockTTSClient
from wavelength_voice.ai_service.contracts import AITurnRequest, PersonaConfig


def test_orchestrator_initialization_and_audio_turn():
    async def _run():
        orchestrator = ConversationOrchestrator(use_mock_providers=True)

        ctx, opening = await orchestrator.initialize_session(
            session_id="s_orch_1",
            user_id="u_orch_1",
            mode="professional",
            scenario_title="Salary Negotiation",
            scenario_description="Negotiating 15% raise",
            persona_name="David Miller",
            persona_role="VP of Engineering",
            difficulty="hard",
        )

        assert ctx.session_id == "s_orch_1"
        assert len(opening) > 0

        state = orchestrator.get_or_create_state("s_orch_1", "u_orch_1")
        assert state.state == SessionState.ACTIVE
        assert len(state.conversation_history) == 1

        # Process user audio turn
        fake_audio = b"\x00" * 16000 * 2
        output, qwen_eval = await orchestrator.process_turn_audio(fake_audio)

        assert output.reply_text is not None
        assert len(output.reply_text) > 0
        assert qwen_eval.scores.overall_turn_score >= 0
        assert len(orchestrator.turn_records) == 1
        assert len(state.conversation_history) == 3  # opening + user + reply

    asyncio.run(_run())


def test_orchestrator_ai_turn_request_wire_contract():
    async def _run():
        orchestrator = ConversationOrchestrator(use_mock_providers=True)

        request = AITurnRequest(
            session_id="s_wire_1",
            user_id="u_wire_1",
            persona=PersonaConfig(
                persona_id="p1",
                name="Alex",
                scenario_prompt="Project review stakeholder",
                difficulty="medium",
                traits=["direct"],
            ),
            transcript="I increased team velocity by 25 percent this quarter.",
            turn_number=1,
        )

        response = await orchestrator.handle_ai_turn_request(request)

        assert response.reply_text is not None
        assert response.end_session is False
        assert response.latency_ms is not None

    asyncio.run(_run())


def test_tts_streaming_and_barge_in():
    async def _run():
        tts = MockTTSClient()
        chunks: list[bytes] = []

        async def _stream():
            async for chunk in tts.stream_audio_response("This is a long sentence being spoken by the AI persona"):
                chunks.append(chunk)

        task = asyncio.create_task(_stream())
        await asyncio.sleep(0.06)
        tts.trigger_interruption()  # Barge-in
        await task

        # Stream was cut off before all words were delivered
        assert len(chunks) < 11

    asyncio.run(_run())


if __name__ == "__main__":
    test_orchestrator_initialization_and_audio_turn()
    test_orchestrator_ai_turn_request_wire_contract()
    test_tts_streaming_and_barge_in()
    print("All conversation orchestrator integration tests passed!")
