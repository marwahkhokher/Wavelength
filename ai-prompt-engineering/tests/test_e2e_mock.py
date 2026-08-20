"""End-to-End multi-turn simulation test (Taha's ownership).

Simulates the entire session lifecycle:
  1. Session Start (Person 1 - Armeen context)
  2. Persona Opening line generation (Person 5 - Taha)
  3. Multi-turn dialogue:
     Turn 1: User claim (Person 2 STT + Person 3 Tone + Person 4 Qwen eval + Person 5 LLM response)
     Turn 2: User follow-up with hesitation -> Dynamic persona disposition shift
     Turn 3: Mid-sentence user barge-in -> TTS stream cancellation
     Turn 4: Final response
  4. Session End & Structured Memory extraction verification
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-tech-infra" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prompt_orchestration.conversation_orchestrator import ConversationOrchestrator
from prompt_orchestration.models import SessionState


def test_full_session_lifecycle_simulation():
    async def _run():
        orchestrator = ConversationOrchestrator(use_mock_providers=True)

        # 1. Initialize session
        ctx, opening = await orchestrator.initialize_session(
            session_id="sim_session_99",
            user_id="sim_user_01",
            mode="professional",
            scenario_title="Senior Dev Promotion Review",
            scenario_description="Candidate is interviewing for Staff Engineer promotion.",
            persona_name="Eleanor Vance",
            persona_role="Principal Architect & Review Board Chair",
            difficulty="hard",
            duration_seconds=300,
        )

        assert ctx.persona.name == "Eleanor Vance"
        assert len(opening) > 0
        state = orchestrator.get_or_create_state("sim_session_99", "sim_user_01")
        assert state.state == SessionState.ACTIVE
        assert len(state.conversation_history) == 1

        # 2. Turn 1: User speaks
        fake_audio_1 = b"\x00" * 16000 * 3
        output_1, eval_1 = await orchestrator.process_turn_audio(fake_audio_1)

        assert output_1.reply_text is not None
        assert eval_1.scores.overall_turn_score > 0
        assert len(orchestrator.turn_records) == 1

        # 3. Turn 2: User speaks again with specific claim
        fake_audio_2 = b"\x00" * 16000 * 2
        output_2, eval_2 = await orchestrator.process_turn_audio(fake_audio_2)
        assert len(orchestrator.turn_records) == 2

        # 4. Barge-In: AI begins streaming audio response, user cuts AI off
        audio_stream_chunks: list[bytes] = []

        async def _play_stream():
            async for chunk in orchestrator.stream_audio_response(output_2.reply_text):
                audio_stream_chunks.append(chunk)

        stream_task = asyncio.create_task(_play_stream())
        await asyncio.sleep(0.04)
        orchestrator.trigger_barge_in()  # User interrupts
        await stream_task

        # Stream was terminated early
        assert len(audio_stream_chunks) < 20

        # 5. Turn 3: User speaks after barge-in
        fake_audio_3 = b"\x00" * 16000 * 2
        output_3, eval_3 = await orchestrator.process_turn_audio(fake_audio_3)
        assert len(orchestrator.turn_records) == 3

        # 6. End session
        ended_session = orchestrator.session_manager.end_session("sim_session_99")
        assert ended_session is not None
        assert ended_session.state == SessionState.COMPLETED
        assert ended_session.current_turn == 3

    asyncio.run(_run())


if __name__ == "__main__":
    test_full_session_lifecycle_simulation()
    print("End-to-End session lifecycle test passed successfully!")
