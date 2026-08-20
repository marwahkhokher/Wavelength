"""Option A — direct STT + filler test with your own voice recording.

Requires a WAV file at demos/test_voice.wav (16-bit PCM, mono, 16 kHz).

Run:
    python demos/test_my_voice.py
"""

from __future__ import annotations

import asyncio
import io
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "voice-tech-infra" / "src"))
sys.path.insert(0, str(REPO_ROOT / "ai-prompt-engineering"))

WAV_PATH = Path(__file__).with_name("test_voice.wav")


def debug_wav(audio_bytes: bytes) -> None:
    print("--- WAV diagnostics ---")
    try:
        import wave

        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            print(
                f"wave module: channels={wf.getnchannels()}, "
                f"sampwidth={wf.getsampwidth()}, "
                f"framerate={wf.getframerate()}, "
                f"frames={wf.getnframes()}, "
                f"duration={round(wf.getnframes() / wf.getframerate(), 2)}s"
            )
    except Exception as exc:
        print(f"wave module failed: {exc}")

    try:
        import soundfile as sf

        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        print(
            f"soundfile: shape={getattr(audio, 'shape', '?')}, "
            f"sr={sr}, "
            f"min={float(audio.min()):.6f}, "
            f"max={float(audio.max()):.6f}, "
            f"mean={float(audio.mean()):.6f}"
        )
    except ImportError:
        print("soundfile: not installed")
    except Exception as exc:
        print(f"soundfile failed: {exc}")

    print("--- end diagnostics ---\n")


async def main() -> None:
    if not WAV_PATH.exists():
        print(f"Missing WAV file: {WAV_PATH}")
        print("Record ~5s of audio and save it there as 16-bit PCM mono 16kHz WAV.")
        sys.exit(1)

    audio_bytes = WAV_PATH.read_bytes()
    print(f"Loaded {len(audio_bytes):,} bytes from {WAV_PATH}\n")
    debug_wav(audio_bytes)

    # --- Streaming path (what the live server uses) ---
    print("=" * 60)
    print("  Streaming STT (WhisperSTTStream)")
    print("=" * 60)
    stream = WhisperSTTStream(model_size="small", language="auto")
    await stream.start()
    await stream.send_audio(audio_bytes)
    await stream.finish()

    async for event in stream.events():
        if event.type == "transcript":
            print(f"transcript : {event.text}")
            print(f"is_final   : {event.is_final}")

    # --- Batch path + filler detection ---
    print("\n" + "=" * 60)
    print("  Batch STT + Filler Detection")
    print("=" * 60)
    result = await WhisperSTTEngine().transcribe_audio_bytes(audio_bytes, sample_rate=16000)
    fillers = FillerDetector().detect_fillers(result.transcript, result.word_timestamps)

    print(f"transcript            : {result.transcript}")
    print(f"total_words           : {result.total_words}")
    print(f"utterance_duration_sec: {result.utterance_duration_sec}")
    print(f"stt_latency_ms        : {result.stt_latency_ms:.2f}")
    print(f"filler_word_count     : {result.filler_word_count}")
    print(f"filler_words          : {[f.word for f in fillers]}")
    print(f"filler_timestamps     : {[(f.word, round(f.start_time, 2), round(f.end_time, 2)) for f in fillers]}")


if __name__ == "__main__":
    from wavelength_voice.voice_pipeline.stt import WhisperSTTStream
    from stt_filler.stt_engine import WhisperSTTEngine
    from stt_filler.filler_detector import FillerDetector

    asyncio.run(main())
