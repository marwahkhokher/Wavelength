"""Streaming text-to-speech: an abstract interface plus an ElevenLabs implementation.

``cancel()`` exists specifically for barge-in: when the turn-taking
controller reports the user interrupted the agent, the orchestrator calls
``cancel()`` so audio generation stops and no further chunks are sent to a
client that has already stopped listening.
"""

from __future__ import annotations

import abc
import asyncio
import base64
import json
from collections.abc import AsyncIterator
from typing import Any

import websockets


class TTSStream(abc.ABC):
    """Interface for synthesizing one agent utterance as a stream of audio chunks."""

    @abc.abstractmethod
    def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Yield raw audio chunks for ``text`` as they become available."""

    @abc.abstractmethod
    async def cancel(self) -> None:
        """Stop generation/playback immediately (barge-in). Idempotent."""


class ElevenLabsTTSStream(TTSStream):
    """Wraps ElevenLabs' websocket streaming TTS API.

    Protocol reference: https://elevenlabs.io/docs/api-reference/websockets
    """

    _BASE_URL = "wss://api.elevenlabs.io/v1/text-to-speech"

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        model_id: str = "eleven_turbo_v2_5",
        output_format: str = "pcm_16000",
    ) -> None:
        self._api_key = api_key
        self._voice_id = voice_id
        self._model_id = model_id
        self._output_format = output_format
        self._cancelled = asyncio.Event()
        self._ws: websockets.WebSocketClientProtocol | None = None

    def _connect_url(self) -> str:
        return (
            f"{self._BASE_URL}/{self._voice_id}/stream-input"
            f"?model_id={self._model_id}&output_format={self._output_format}"
        )

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        self._cancelled.clear()
        async with websockets.connect(self._connect_url()) as ws:
            self._ws = ws
            await ws.send(
                json.dumps(
                    {
                        "text": " ",
                        "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
                        "xi_api_key": self._api_key,
                    }
                )
            )
            await ws.send(json.dumps({"text": text, "try_trigger_generation": True}))
            await ws.send(json.dumps({"text": ""}))  # signal end of input

            try:
                async for raw in ws:
                    if self._cancelled.is_set():
                        break
                    chunk = self._parse_message(raw)
                    if chunk is not None:
                        yield chunk
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                self._ws = None

    async def cancel(self) -> None:
        self._cancelled.set()
        if self._ws is not None:
            await self._ws.close()

    @staticmethod
    def _parse_message(raw: str | bytes) -> bytes | None:
        try:
            message: dict[str, Any] = json.loads(raw)
        except (TypeError, ValueError):
            return None
        audio_b64 = message.get("audio")
        if not audio_b64:
            return None
        return base64.b64decode(audio_b64)
