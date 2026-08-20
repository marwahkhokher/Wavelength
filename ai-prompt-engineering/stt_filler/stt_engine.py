"""Whisper Small STT Engine (Areej's ownership).

Handles audio transcription, word counts, and STTResult production.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-tech-infra" / "src"))

from wavelength_voice.ai_service.contracts import STTResult
from .filler_detector import FillerDetector


class WhisperSTTEngine:
    """WhisperSmall Speech-to-Text transcriber and utterance analyzer."""

    def __init__(self, model_size: str = "small"):
        self.model_size = model_size
        self.filler_detector = FillerDetector()

    async def transcribe_audio_bytes(
        self, audio_bytes: bytes, sample_rate: int = 16000
    ) -> STTResult:
        """Transcribes incoming audio bytes and extracts filler words."""
        t0 = time.perf_counter()
        
        # Stub/Production transcriber interface
        # When real Whisper model is loaded, swap in whisper.transcribe(audio_bytes)
        simulated_transcript = (
            "Um, I believe my contribution to the project, like, increased team velocity."
        )
        duration_sec = max(1.0, len(audio_bytes) / (sample_rate * 2))
        
        fillers = self.filler_detector.detect_fillers(simulated_transcript)
        words = simulated_transcript.split()
        
        latency_ms = (time.perf_counter() - t0) * 1000
        
        return STTResult(
            transcript=simulated_transcript,
            is_final=True,
            total_words=len(words),
            filler_word_count=len(fillers),
            filler_words=fillers,
            utterance_duration_sec=duration_sec,
            stt_latency_ms=latency_ms,
        )
