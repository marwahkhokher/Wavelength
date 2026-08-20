"""Tests for Areej's STT + Filler modules (Step 1 of the AI pipeline)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-tech-infra" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ai-prompt-engineering"))

from wavelength_voice.ai_service.contracts import WordTimestamp
from stt_filler.filler_detector import FillerDetector
from stt_filler.stt_engine import WhisperSTTEngine


def _fake_whisper_result():
    return {
        "text": "Um I believe my contribution to the project like increased team velocity",
        "segments": [
            {"start": 0.0, "end": 0.45, "text": "Um"},
            {"start": 0.5, "end": 0.65, "text": "I"},
            {"start": 0.7, "end": 1.1, "text": "believe"},
            {"start": 1.15, "end": 1.35, "text": "my"},
            {"start": 1.4, "end": 1.95, "text": "contribution"},
            {"start": 2.0, "end": 2.15, "text": "to"},
            {"start": 2.2, "end": 2.35, "text": "the"},
            {"start": 2.4, "end": 2.9, "text": "project"},
            {"start": 3.0, "end": 3.35, "text": "like"},
            {"start": 3.4, "end": 3.85, "text": "increased"},
            {"start": 3.9, "end": 4.2, "text": "team"},
            {"start": 4.25, "end": 4.8, "text": "velocity"},
        ],
    }


class FakeWhisperModel:
    def transcribe(self, audio, language=None, word_timestamps=False):
        return _fake_whisper_result()


def test_filler_detection_with_timestamps():
    detector = FillerDetector()
    timestamps = [
        WordTimestamp(word="Um", start=0.0, end=0.45),
        WordTimestamp(word="I", start=0.5, end=0.65),
        WordTimestamp(word="think", start=0.7, end=1.0),
        WordTimestamp(word="like", start=1.1, end=1.45),
        WordTimestamp(word="we", start=1.5, end=1.65),
        WordTimestamp(word="should", start=1.7, end=2.05),
        WordTimestamp(word="start", start=2.1, end=2.5),
    ]
    fillers = detector.detect_fillers("", word_timestamps=timestamps)
    assert len(fillers) == 2
    assert fillers[0].word == "um"
    assert fillers[0].start_time == 0.0
    assert fillers[0].end_time == 0.45
    assert fillers[1].word == "like"
    assert fillers[1].start_time == 1.1


def test_filler_detection_roman_urdu_with_timestamps():
    detector = FillerDetector()
    timestamps = [
        WordTimestamp(word="Matlab", start=0.0, end=0.5),
        WordTimestamp(word="yaar", start=0.6, end=1.0),
        WordTimestamp(word="I", start=1.1, end=1.3),
        WordTimestamp(word="think", start=1.4, end=1.7),
        WordTimestamp(word="bas", start=1.8, end=2.1),
        WordTimestamp(word="achha", start=2.2, end=2.6),
    ]
    fillers = detector.detect_fillers("", word_timestamps=timestamps)
    assert len(fillers) >= 3
    words = {f.word for f in fillers}
    assert "matlab" in words
    assert "yaar" in words
    assert "bas" in words
    assert "achha" in words


def test_filler_detection_fallback():
    detector = FillerDetector()
    fillers = detector.detect_fillers("Um, I think like we should start.")
    assert len(fillers) == 2
    assert fillers[0].word == "um"
    assert fillers[1].word == "like"


def test_transcribe_returns_stt_result_schema():
    engine = WhisperSTTEngine()
    fake_audio = b"\x00" * (16000 * 2)  # 1 second of 16-bit mono silence

    with patch("whisper.load_model", return_value=FakeWhisperModel()):
        result = asyncio.run(engine.transcribe_audio_bytes(fake_audio, sample_rate=16000))

    assert result.is_final is True
    assert result.total_words > 0
    assert result.filler_word_count >= 0
    assert isinstance(result.filler_words, list)
    assert isinstance(result.word_timestamps, list)
    assert result.utterance_duration_sec > 0
    assert result.stt_latency_ms >= 0


def test_word_timestamps_populated():
    engine = WhisperSTTEngine()
    fake_audio = b"\x00" * (16000 * 2)

    with patch("whisper.load_model", return_value=FakeWhisperModel()):
        result = asyncio.run(engine.transcribe_audio_bytes(fake_audio, sample_rate=16000))

    assert len(result.word_timestamps) > 0
    for ts in result.word_timestamps:
        assert ts.start >= 0.0
        assert ts.end > ts.start


def test_stt_latency_measured():
    engine = WhisperSTTEngine()
    fake_audio = b"\x00" * (16000 * 2)

    with patch("whisper.load_model", return_value=FakeWhisperModel()):
        result = asyncio.run(engine.transcribe_audio_bytes(fake_audio, sample_rate=16000))

    assert result.stt_latency_ms >= 0


if __name__ == "__main__":
    test_filler_detection_with_timestamps()
    test_filler_detection_roman_urdu_with_timestamps()
    test_filler_detection_fallback()
    test_transcribe_returns_stt_result_schema()
    test_word_timestamps_populated()
    test_stt_latency_measured()
    print("Areej STT + Filler tests passed.")
