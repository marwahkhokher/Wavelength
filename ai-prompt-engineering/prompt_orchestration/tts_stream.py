"""ElevenLabs Streaming TTS & Interruption Handler (Taha's ownership).

Provides:
  1. ElevenLabsTTSClient: Real ElevenLabs WebSocket/REST streaming TTS with instant cancellation
  2. MockTTSClient: Deterministic in-memory chunk streamer for tests & offline dev
"""

from __future__ import annotations

import abc
import asyncio
import base64
import json
import logging
import os
from typing import AsyncGenerator

from .config import get_ai_settings

logger = logging.getLogger(__name__)


class BaseTTSClient(abc.ABC):
    """Abstract interface for Streaming TTS clients."""

    @abc.abstractmethod
    def trigger_interruption(self) -> None:
        """Called when VAD detects user speech mid-AI utterance ('Cut AI off')."""
        raise NotImplementedError

    @abc.abstractmethod
    async def stream_audio_response(self, text: str) -> AsyncGenerator[bytes, None]:
        """Streams audio chunks to client, stopping immediately if interrupted."""
        raise NotImplementedError


class MockTTSClient(BaseTTSClient):
    """Deterministic, mock streaming TTS client for testing and offline development."""

    def __init__(self, voice_id: str = "21m00Tcm4TlvDq8ikWAM") -> None:
        self.voice_id = voice_id
        self._is_interrupted = False

    def trigger_interruption(self) -> None:
        """Instantly abort playback stream."""
        self._is_interrupted = True

    async def stream_audio_response(self, text: str) -> AsyncGenerator[bytes, None]:
        self._is_interrupted = False
        words = text.split()

        for word in words:
            if self._is_interrupted:
                logger.info("Mock TTS stream interrupted / cancelled by user barge-in.")
                break

            # Simulated PCM/Audio chunk
            chunk = f"[AudioChunk: {word}]".encode("utf-8")
            yield chunk
            await asyncio.sleep(0.04)


class ElevenLabsTTSClient(BaseTTSClient):
    """ElevenLabs streaming TTS wrapper with VAD interruption handler.

    Uses ElevenLabs WebSocket streaming API when API key is available,
    otherwise gracefully falls back to simulated audio chunk stream.
    """

    _BASE_URL = "wss://api.elevenlabs.io/v1/text-to-speech"

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str | None = None,
        model_id: str | None = None,
        output_format: str | None = None,
    ) -> None:
        settings = get_ai_settings()
        self.api_key = api_key or settings.elevenlabs_api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        self.voice_id = voice_id or settings.elevenlabs_voice_id
        self.model_id = model_id or settings.elevenlabs_model_id
        self.output_format = output_format or settings.elevenlabs_output_format
        self._is_interrupted = False
        self._mock_fallback = MockTTSClient(voice_id=self.voice_id)

    def trigger_interruption(self) -> None:
        """Called when VAD detects user speech mid-AI utterance ('Cut AI off' FR-SESS-4)."""
        self._is_interrupted = True
        self._mock_fallback.trigger_interruption()

    def _connect_url(self) -> str:
        return (
            f"{self._BASE_URL}/{self.voice_id}/stream-input"
            f"?model_id={self.model_id}&output_format={self.output_format}"
        )

    async def stream_audio_response(self, text: str) -> AsyncGenerator[bytes, None]:
        """Streams audio chunks from ElevenLabs WS API or fallback if key not configured."""
        if not self.api_key:
            async for chunk in self._mock_fallback.stream_audio_response(text):
                yield chunk
            return

        self._is_interrupted = False
        try:
            import websockets
        except ImportError:
            logger.warning("websockets not installed, falling back to mock TTS stream")
            async for chunk in self._mock_fallback.stream_audio_response(text):
                yield chunk
            return

        try:
            async with websockets.connect(self._connect_url()) as ws:
                # 1. Send initial handshake with API key
                await ws.send(
                    json.dumps(
                        {
                            "text": " ",
                            "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
                            "xi_api_key": self.api_key,
                        }
                    )
                )
                # 2. Send text payload
                await ws.send(json.dumps({"text": text, "try_trigger_generation": True}))
                await ws.send(json.dumps({"text": ""}))  # End of input signal

                # 3. Stream incoming audio chunks
                async for raw in ws:
                    if self._is_interrupted:
                        logger.info("ElevenLabs TTS stream interrupted by barge-in")
                        break

                    try:
                        msg = json.loads(raw)
                        audio_b64 = msg.get("audio")
                        if audio_b64:
                            yield base64.b64decode(audio_b64)
                        if msg.get("isFinal"):
                            break
                    except Exception:
                        continue
        except Exception as e:
            logger.warning("ElevenLabs connection error (%s), falling back to mock stream", e)
            async for chunk in self._mock_fallback.stream_audio_response(text):
                yield chunk
