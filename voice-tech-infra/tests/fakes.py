"""In-memory fakes for STTStream/TTSStream used to test the orchestrator
without a network dependency on Deepgram/ElevenLabs.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from wavelength_voice.voice_pipeline.stt import STTEvent, STTStream
from wavelength_voice.voice_pipeline.tts import TTSStream


class FakeSTTStream(STTStream):
    def __init__(self) -> None:
        self._queue: asyncio.Queue[STTEvent] = asyncio.Queue()
        self.sent_audio: list[bytes] = []
        self.started = False
        self.finished = False

    async def start(self) -> None:
        self.started = True

    async def send_audio(self, chunk: bytes) -> None:
        self.sent_audio.append(chunk)

    async def finish(self) -> None:
        self.finished = True

    async def events(self) -> AsyncIterator[STTEvent]:
        while True:
            event = await self._queue.get()
            yield event
            if event.type in ("closed", "error"):
                return

    async def emit(self, event: STTEvent) -> None:
        await self._queue.put(event)


class FakeTTSStream(TTSStream):
    def __init__(self, num_chunks: int = 5, chunk_delay: float = 0.02) -> None:
        self._num_chunks = num_chunks
        self._chunk_delay = chunk_delay
        self.cancelled = False
        self.synthesized_texts: list[str] = []

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        self.synthesized_texts.append(text)
        for i in range(self._num_chunks):
            if self.cancelled:
                return
            await asyncio.sleep(self._chunk_delay)
            if self.cancelled:
                return
            yield f"chunk-{i}".encode()

    async def cancel(self) -> None:
        self.cancelled = True
