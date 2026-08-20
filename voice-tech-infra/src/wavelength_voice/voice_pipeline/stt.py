"""Streaming speech-to-text: an abstract interface plus a Deepgram implementation.

Everything above this module (the orchestrator, turn-taking) talks to the
``STTStream`` interface and ``STTEvent`` stream, never to Deepgram directly.
That keeps the orchestrator testable with a fake STT stream and means a
different STT vendor could be swapped in without touching turn-taking or
session logic.
"""

from __future__ import annotations

import abc
import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

import websockets

STTEventType = Literal["speech_started", "transcript", "utterance_end", "error", "closed"]


@dataclass
class STTEvent:
    type: STTEventType
    text: str | None = None
    is_final: bool = False
    confidence: float | None = None
    message: str | None = None


class STTStream(abc.ABC):
    """Interface for a single streaming STT session (one per connected user)."""

    @abc.abstractmethod
    async def start(self) -> None:
        """Open the upstream connection."""

    @abc.abstractmethod
    async def send_audio(self, chunk: bytes) -> None:
        """Push a raw audio chunk (matching the configured encoding/sample rate)."""

    @abc.abstractmethod
    async def finish(self) -> None:
        """Signal end-of-audio and release resources."""

    @abc.abstractmethod
    def events(self) -> AsyncIterator[STTEvent]:
        """Yield STT events (speech_started, interim/final transcripts, ...) as they arrive."""


class DeepgramSTTStream(STTStream):
    """Wraps Deepgram's real-time streaming transcription websocket API.

    Protocol reference: https://developers.deepgram.com/docs/streaming
    """

    _BASE_URL = "wss://api.deepgram.com/v1/listen"

    def __init__(
        self,
        api_key: str,
        model: str = "nova-2",
        language: str = "en-US",
        sample_rate: int = 16000,
        encoding: str = "linear16",
        interim_results: bool = True,
        vad_events: bool = True,
        endpointing_ms: int = 300,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._language = language
        self._sample_rate = sample_rate
        self._encoding = encoding
        self._interim_results = interim_results
        self._vad_events = vad_events
        self._endpointing_ms = endpointing_ms

        self._ws: websockets.WebSocketClientProtocol | None = None
        self._queue: asyncio.Queue[STTEvent] = asyncio.Queue()
        self._receive_task: asyncio.Task[None] | None = None

    def _connect_url(self) -> str:
        params = {
            "model": self._model,
            "language": self._language,
            "sample_rate": str(self._sample_rate),
            "encoding": self._encoding,
            "interim_results": str(self._interim_results).lower(),
            "vad_events": str(self._vad_events).lower(),
            "endpointing": str(self._endpointing_ms),
        }
        query = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self._BASE_URL}?{query}"

    async def start(self) -> None:
        self._ws = await websockets.connect(
            self._connect_url(),
            extra_headers={"Authorization": f"Token {self._api_key}"},
        )
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def send_audio(self, chunk: bytes) -> None:
        if self._ws is None:
            raise RuntimeError("DeepgramSTTStream.start() must be called before send_audio()")
        await self._ws.send(chunk)

    async def finish(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.send(json.dumps({"type": "CloseStream"}))
            except websockets.exceptions.ConnectionClosed:
                pass
        if self._receive_task is not None:
            await self._receive_task
        if self._ws is not None:
            await self._ws.close()

    async def events(self) -> AsyncIterator[STTEvent]:
        while True:
            event = await self._queue.get()
            yield event
            if event.type in ("closed", "error"):
                return

    async def _receive_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                event = self._parse_message(raw)
                if event is not None:
                    await self._queue.put(event)
        except websockets.exceptions.ConnectionClosedOK:
            await self._queue.put(STTEvent(type="closed"))
        except websockets.exceptions.ConnectionClosedError as exc:
            await self._queue.put(STTEvent(type="error", message=str(exc)))
        except Exception as exc:  # noqa: BLE001 - surface any decode/protocol error upstream
            await self._queue.put(STTEvent(type="error", message=str(exc)))

    @staticmethod
    def _parse_message(raw: str | bytes) -> STTEvent | None:
        try:
            message: dict[str, Any] = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return None

        msg_type = message.get("type")
        if msg_type == "SpeechStarted":
            return STTEvent(type="speech_started")
        if msg_type == "UtteranceEnd":
            return STTEvent(type="utterance_end")
        if msg_type == "Results":
            channel = message.get("channel", {})
            alternatives = channel.get("alternatives", [])
            if not alternatives:
                return None
            transcript = alternatives[0].get("transcript", "")
            if not transcript:
                return None
            return STTEvent(
                type="transcript",
                text=transcript,
                is_final=bool(message.get("is_final", False)),
                confidence=alternatives[0].get("confidence"),
            )
        return None
