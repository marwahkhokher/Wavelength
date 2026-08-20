"""Tests for the Whisper Small streaming STT backend."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import numpy as np
import pytest

from wavelength_voice.voice_pipeline.stt import STTEvent, WhisperSTTStream


def _fake_whisper_result(text: str = "hello world"):
    return {
        "text": text,
        "segments": [
            {"start": 0.0, "end": 0.5, "text": "hello"},
            {"start": 0.5, "end": 1.0, "text": "world"},
        ],
    }


class FakeWhisperModel:
    def transcribe(self, audio, language=None, word_timestamps=False):
        return _fake_whisper_result()


@pytest.mark.asyncio
async def test_whisper_stream_emits_expected_events():
    with patch("whisper.load_model", return_value=FakeWhisperModel()):
        stream = WhisperSTTStream(model_size="small", language="auto")
        await stream.start()
        await stream.send_audio(b"\x00" * 16000 * 2)  # 1s silence
        await stream.finish()

        events = [e async for e in stream.events()]
        types = [e.type for e in events]

        assert "speech_started" in types
        assert "transcript" in types
        assert "utterance_end" in types
        assert "closed" in types


@pytest.mark.asyncio
async def test_whisper_stream_transcript_is_final():
    with patch("whisper.load_model", return_value=FakeWhisperModel()):
        stream = WhisperSTTStream(model_size="small", language="auto")
        await stream.start()
        await stream.send_audio(b"\x00" * 16000 * 2)
        await stream.finish()

        events = [e async for e in stream.events()]
        transcript_events = [e for e in events if e.type == "transcript"]
        assert len(transcript_events) == 1
        assert transcript_events[0].is_final is True
        assert transcript_events[0].text


@pytest.mark.asyncio
async def test_whisper_stream_started_implicitly_on_send():
    with patch("whisper.load_model", return_value=FakeWhisperModel()):
        stream = WhisperSTTStream(model_size="small", language="auto")
        await stream.send_audio(b"\x00" * 16000 * 2)
        await stream.finish()

        events = [e async for e in stream.events()]
        types = [e.type for e in events]
        assert "speech_started" in types
        assert "transcript" in types


@pytest.mark.asyncio
async def test_whisper_stream_empty_audio_still_completes():
    stream = WhisperSTTStream(model_size="small", language="auto")
    await stream.start()
    await stream.finish()

    events = [e async for e in stream.events()]
    types = [e.type for e in events]
    assert "utterance_end" in types
    assert "closed" in types
    transcript_events = [e for e in events if e.type == "transcript"]
    assert len(transcript_events) == 0


@pytest.mark.asyncio
async def test_whisper_stream_language_param_passed_to_model():
    captured = {}

    class CapturingModel:
        def transcribe(self, audio, language=None, word_timestamps=False):
            captured["language"] = language
            return _fake_whisper_result()

    with patch("whisper.load_model", return_value=CapturingModel()):
        stream = WhisperSTTStream(model_size="small", language="en")
        await stream.start()
        await stream.send_audio(b"\x00" * 16000 * 2)
        await stream.finish()

        events = [e async for e in stream.events()]
        transcript_events = [e for e in events if e.type == "transcript"]
        assert len(transcript_events) == 1
        assert captured["language"] == "en"


@pytest.mark.asyncio
async def test_whisper_stream_multiple_chunks_accumulated():
    with patch("whisper.load_model", return_value=FakeWhisperModel()):
        stream = WhisperSTTStream(model_size="small", language="auto")
        await stream.start()
        for _ in range(3):
            await stream.send_audio(b"\x00" * 16000 * 2)
        await stream.finish()

        events = [e async for e in stream.events()]
        types = [e.type for e in events]
        assert "transcript" in types
        assert "closed" in types
