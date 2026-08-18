"""End-to-end orchestrator tests: STT -> turn-taking -> mock AI -> TTS,
including barge-in cancelling an in-flight TTS stream mid-playback.
"""

from __future__ import annotations

import asyncio

from wavelength_voice.ai_service.client import MockAIServiceClient
from wavelength_voice.ai_service.contracts import PersonaConfig
from wavelength_voice.session_state.manager import SessionManager
from wavelength_voice.voice_pipeline.stt import STTEvent
from wavelength_voice.voice_pipeline.turn_taking import TurnState
from wavelength_voice.ws.orchestrator import SessionOrchestrator

from .fakes import FakeSTTStream, FakeTTSStream

PERSONA = PersonaConfig(persona_id="p1", name="Alex", scenario_prompt="Test persona")


async def make_orchestrator(
    session_manager: SessionManager,
    stt: FakeSTTStream,
    tts: FakeTTSStream,
    ai_client: MockAIServiceClient,
    session_id: str = "s1",
):
    await session_manager.connect(session_id, "conn-a", user_id="u1")
    sent_audio: list[bytes] = []
    sent_events: list[dict] = []

    async def send_audio(chunk: bytes) -> None:
        sent_audio.append(chunk)

    async def send_event(event: dict) -> None:
        sent_events.append(event)

    orchestrator = SessionOrchestrator(
        session_id=session_id,
        persona=PERSONA,
        stt=stt,
        tts=tts,
        ai_client=ai_client,
        session_manager=session_manager,
        send_audio=send_audio,
        send_event=send_event,
    )
    await orchestrator.start()
    return orchestrator, sent_audio, sent_events


async def test_full_turn_streams_all_tts_audio_and_returns_to_waiting() -> None:
    session_manager = SessionManager()
    stt = FakeSTTStream()
    tts = FakeTTSStream(num_chunks=3, chunk_delay=0.01)
    ai_client = MockAIServiceClient(replies=["Sure, tell me more."])

    orchestrator, sent_audio, sent_events = await make_orchestrator(
        session_manager, stt, tts, ai_client
    )

    await stt.emit(STTEvent(type="speech_started"))
    await stt.emit(STTEvent(type="transcript", text="hello there", is_final=True))

    # Wait for the AI call + full TTS stream to complete.
    for _ in range(50):
        await asyncio.sleep(0.01)
        if orchestrator.turns.state == TurnState.WAITING_FOR_USER:
            break

    assert orchestrator.turns.state == TurnState.WAITING_FOR_USER
    assert len(sent_audio) == 3
    assert tts.synthesized_texts == ["Sure, tell me more."]
    session = session_manager.get_session("s1")
    assert [t.text for t in session.conversation_history] == [
        "hello there",
        "Sure, tell me more.",
    ]
    assert {"type": "agent_reply_text", "text": "Sure, tell me more."} in sent_events
    assert {"type": "agent_speech_ended"} in sent_events

    await orchestrator.stop()


async def test_final_transcript_without_prior_speech_started_still_completes_turn() -> None:
    """Some STT providers can drop/coalesce the interim speech_started event;
    the orchestrator must treat a final transcript as an implicit turn-start
    rather than raising and silently killing the STT-consumer task."""
    session_manager = SessionManager()
    stt = FakeSTTStream()
    tts = FakeTTSStream(num_chunks=1, chunk_delay=0.01)
    ai_client = MockAIServiceClient(replies=["got it"])

    orchestrator, sent_audio, sent_events = await make_orchestrator(
        session_manager, stt, tts, ai_client
    )

    await stt.emit(STTEvent(type="transcript", text="no speech_started first", is_final=True))

    for _ in range(50):
        await asyncio.sleep(0.01)
        if orchestrator.turns.state == TurnState.WAITING_FOR_USER:
            break

    assert orchestrator.turns.state == TurnState.WAITING_FOR_USER
    assert len(sent_audio) == 1
    assert tts.synthesized_texts == ["got it"]

    await orchestrator.stop()


async def test_barge_in_mid_playback_cancels_tts_and_stops_audio() -> None:
    session_manager = SessionManager()
    stt = FakeSTTStream()
    tts = FakeTTSStream(num_chunks=10, chunk_delay=0.03)
    ai_client = MockAIServiceClient(replies=["This is a long reply that keeps going."])

    orchestrator, sent_audio, sent_events = await make_orchestrator(
        session_manager, stt, tts, ai_client
    )

    await stt.emit(STTEvent(type="speech_started"))
    await stt.emit(STTEvent(type="transcript", text="what's the plan", is_final=True))

    # Let the agent start speaking and stream a couple of chunks.
    for _ in range(50):
        await asyncio.sleep(0.01)
        if orchestrator.turns.state == TurnState.AGENT_SPEAKING and sent_audio:
            break
    assert orchestrator.turns.state == TurnState.AGENT_SPEAKING

    chunks_before_barge_in = len(sent_audio)
    assert 0 < chunks_before_barge_in < 10

    # User interrupts mid-playback.
    await stt.emit(STTEvent(type="speech_started"))
    await asyncio.sleep(0.05)  # let cancellation propagate

    assert orchestrator.turns.state == TurnState.USER_SPEAKING
    assert orchestrator.turns.barge_in_count == 1
    assert tts.cancelled is True

    # No more audio should arrive after the interruption settles.
    audio_count_after_settle = len(sent_audio)
    await asyncio.sleep(0.1)
    assert len(sent_audio) == audio_count_after_settle
    assert {"type": "agent_speech_ended"} not in sent_events

    await orchestrator.stop()


async def test_ai_service_still_thinking_when_barged_in_response_is_dropped() -> None:
    session_manager = SessionManager()
    stt = FakeSTTStream()
    tts = FakeTTSStream()
    ai_client = MockAIServiceClient(replies=["slow reply"], latency_seconds=0.1)

    orchestrator, sent_audio, sent_events = await make_orchestrator(
        session_manager, stt, tts, ai_client
    )

    await stt.emit(STTEvent(type="speech_started"))
    await stt.emit(STTEvent(type="transcript", text="hmm", is_final=True))
    await asyncio.sleep(0.02)
    assert orchestrator.turns.state == TurnState.PROCESSING

    # Barge in before the (slow) AI response ever arrives.
    await stt.emit(STTEvent(type="speech_started"))
    await asyncio.sleep(0.02)
    assert orchestrator.turns.state == TurnState.USER_SPEAKING

    # Give the slow AI call time to resolve, if it wasn't cancelled - it must
    # not resurrect a stale agent turn.
    await asyncio.sleep(0.2)

    assert orchestrator.turns.state == TurnState.USER_SPEAKING
    assert sent_audio == []
    assert not any(e.get("type") == "agent_reply_text" for e in sent_events)

    await orchestrator.stop()
