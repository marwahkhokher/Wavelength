"""Rough turn-taking latency benchmark.

Not a pytest suite - a standalone script to eyeball end-to-end turn latency
(user utterance finalized -> first agent audio chunk sent) using the mock AI
service and in-memory fake STT/TTS streams. Once real Deepgram/ElevenLabs
credentials are available, swap in ``DeepgramSTTStream``/``ElevenLabsTTSStream``
(see ``wavelength_voice.voice_pipeline``) to get real network-inclusive numbers.

Run: python latency-tests/turn_latency_benchmark.py
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from wavelength_voice.ai_service.client import MockAIServiceClient  # noqa: E402
from wavelength_voice.ai_service.contracts import PersonaConfig  # noqa: E402
from wavelength_voice.session_state.manager import SessionManager  # noqa: E402
from wavelength_voice.voice_pipeline.stt import STTEvent  # noqa: E402
from wavelength_voice.ws.orchestrator import SessionOrchestrator  # noqa: E402

from fakes import FakeSTTStream, FakeTTSStream  # noqa: E402

PERSONA = PersonaConfig(persona_id="bench", name="Alex", scenario_prompt="Benchmark persona")


async def run_one_turn(ai_latency_seconds: float, tts_chunk_delay: float) -> float:
    session_manager = SessionManager()
    await session_manager.connect("bench-session", "conn-a", user_id="bench-user")

    stt = FakeSTTStream()
    tts = FakeTTSStream(num_chunks=1, chunk_delay=tts_chunk_delay)
    ai_client = MockAIServiceClient(latency_seconds=ai_latency_seconds)

    first_audio_at: list[float] = []

    async def send_audio(chunk: bytes) -> None:
        first_audio_at.append(time.perf_counter())

    async def send_event(event: dict) -> None:
        pass

    orchestrator = SessionOrchestrator(
        session_id="bench-session",
        persona=PERSONA,
        stt=stt,
        tts=tts,
        ai_client=ai_client,
        session_manager=session_manager,
        send_audio=send_audio,
        send_event=send_event,
    )
    await orchestrator.start()

    t0 = time.perf_counter()
    await stt.emit(STTEvent(type="speech_started"))
    await stt.emit(STTEvent(type="transcript", text="hello", is_final=True))

    while not first_audio_at:
        await asyncio.sleep(0.005)

    await orchestrator.stop()
    return first_audio_at[0] - t0


async def main() -> None:
    # Rough stand-ins for expected production latencies (seconds).
    ai_latency_seconds = 0.3
    tts_chunk_delay = 0.15

    samples = [
        await run_one_turn(ai_latency_seconds, tts_chunk_delay) for _ in range(20)
    ]
    samples_ms = sorted(s * 1000 for s in samples)

    print(f"n={len(samples_ms)}")
    print(f"mean:   {statistics.mean(samples_ms):.1f} ms")
    print(f"median: {statistics.median(samples_ms):.1f} ms")
    print(f"p95:    {samples_ms[int(len(samples_ms) * 0.95) - 1]:.1f} ms")
    print(f"max:    {max(samples_ms):.1f} ms")


if __name__ == "__main__":
    asyncio.run(main())
