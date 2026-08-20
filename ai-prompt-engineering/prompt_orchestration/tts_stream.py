"""ElevenLabs Streaming TTS & Interruption Handler (Taha's ownership).

Handles streaming text-to-speech conversion and VAD interruption ('Cut AI off').
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator


class ElevenLabsTTSClient:
    """ElevenLabs streaming TTS wrapper with VAD interruption handler."""

    def __init__(self, voice_id: str = "21m00Tcm4TlvDq8ikWAM"):
        self.voice_id = voice_id
        self._is_interrupted = False

    def trigger_interruption(self) -> None:
        """Called when VAD detects user speech mid-AI utterance ('Cut AI off' FR-SESS-4)."""
        self._is_interrupted = True

    async def stream_audio_response(self, text: str) -> AsyncGenerator[bytes, None]:
        """Streams audio chunks to client, stopping immediately if interrupted."""
        self._is_interrupted = False
        words = text.split()
        
        for word in words:
            if self._is_interrupted:
                # Instantly abort playback stream
                break
                
            # Simulated audio chunk (e.g. 100ms MP3/PCM audio chunk)
            chunk = f"[AudioChunk: {word}]".encode("utf-8")
            yield chunk
            await asyncio.sleep(0.08)
