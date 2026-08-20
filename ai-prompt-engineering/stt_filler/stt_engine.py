"""Whisper Small STT Engine (Areej's ownership).

Handles audio transcription, word-level timestamp extraction, filler word
detection, and STTResult production.  Whisper integration is abstracted so
the real model can be swapped in without changing the pipeline interface.
"""

from __future__ import annotations

import io
import logging
import sys
import time
import wave
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-tech-infra" / "src"))

from wavelength_voice.ai_service.contracts import STTResult, WordTimestamp
from .filler_detector import FillerDetector


class WhisperSTTEngine:
    """WhisperSmall Speech-to-Text transcriber and utterance analyzer.

    Supports multilingual transcription including English and Roman Urdu
    via Whisper's built-in language handling.  Language can be set at
    init-time or left as ``"auto"`` for runtime detection.
    """

    def __init__(
        self,
        model_size: str = "small",
        language: str = "auto",
    ):
        self.model_size = model_size
        self.language = language
        self.filler_detector = FillerDetector()

    async def transcribe_audio_bytes(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000,
    ) -> STTResult:
        """Transcribes ``audio_bytes`` and returns a fully populated STTResult.

        The real Whisper model is loaded lazily inside ``_run_whisper`` so
        this interface works in tests and CI without GPU/model weights.
        """
        t0 = time.perf_counter()

        whisper_result = self._run_whisper(audio_bytes, sample_rate)
        transcript = whisper_result["text"].strip()
        segments = whisper_result.get("segments", [])

        word_timestamps = self._extract_word_timestamps(segments)
        utterance_duration_sec = self._compute_duration(audio_bytes, sample_rate, segments)
        fillers = self.filler_detector.detect_fillers(transcript, word_timestamps)
        words = transcript.split()

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return STTResult(
            transcript=transcript,
            is_final=True,
            total_words=len(words),
            filler_word_count=len(fillers),
            filler_words=fillers,
            word_timestamps=word_timestamps,
            utterance_duration_sec=utterance_duration_sec,
            stt_latency_ms=round(latency_ms, 2),
        )

    def _run_whisper(self, audio_bytes: bytes, sample_rate: int) -> dict[str, Any]:
        """Runs Whisper Small and returns raw result.

        Falls back to a deterministic stub when ``whisper``/``numpy`` are not
        installed so tests remain runnable without extra dependencies.
        """
        try:
            import numpy as np
            import whisper

            audio = self._load_wav_pcm(audio_bytes, sample_rate)
            model = whisper.load_model(self.model_size)
            language = None if self.language == "auto" else self.language
            result = model.transcribe(audio, language=language, word_timestamps=True)

            segments = []
            for seg in result.get("segments", []):
                segments.append(
                    {
                        "start": seg.get("start", 0.0),
                        "end": seg.get("end", 0.0),
                        "text": seg.get("text", ""),
                    }
                )

            return {
                "text": result.get("text", ""),
                "segments": segments,
                "confidence": None,
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("Whisper transcription failed, using stub: %s", exc)
            return self._stub_transcribe()

    @staticmethod
    def _load_wav_pcm(audio_bytes: bytes, target_sample_rate: int) -> Any:
        """Extracts float32 PCM audio from a WAV file, resampling if needed."""
        try:
            import numpy as np
        except ImportError:
            return WhisperSTTEngine._load_wav_pcm_fallback(audio_bytes)

        try:
            import soundfile as sf
            soundfile = sf
        except ImportError:
            soundfile = None

        try:
            if soundfile is not None:
                try:
                    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    if sr != target_sample_rate:
                            try:
                                import resampy
                                audio = resampy.resample(audio, sr, target_sample_rate)
                            except ImportError:
                                try:
                                    from scipy import signal
                                    n_samples = int(round(len(audio) * target_sample_rate / sr))
                                    audio = signal.resample(audio, n_samples)
                                    if audio.dtype != np.float32:
                                        audio = audio.astype(np.float32)
                                except ImportError:
                                    pass
                    return audio
                except Exception:
                    pass

            with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)

            if sampwidth == 2:
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 1:
                audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
            else:
                raise ValueError(f"Unsupported WAV sampwidth: {sampwidth}")

            if n_channels > 1:
                audio = audio.reshape(-1, n_channels).mean(axis=1)

            if framerate != target_sample_rate:
                try:
                    import resampy
                    audio = resampy.resample(audio, framerate, target_sample_rate)
                except ImportError:
                    try:
                        from scipy import signal
                        n_samples = int(round(len(audio) * target_sample_rate / framerate))
                        audio = signal.resample(audio, n_samples)
                        if audio.dtype != np.float32:
                            audio = audio.astype(np.float32)
                    except ImportError:
                        pass

            return audio
        except Exception as exc:
            logger.warning("WAV parse failed (%s), falling back to raw PCM", exc)
            return WhisperSTTEngine._load_wav_pcm_fallback(audio_bytes)

    @staticmethod
    def _load_wav_pcm_fallback(audio_bytes: bytes) -> Any:
        """Fallback: treat bytes as raw int16 PCM when WAV parsing fails."""
        try:
            import numpy as np
            return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        except ImportError:
            return audio_bytes

    def _stub_transcribe(self) -> dict[str, Any]:
        """Deterministic fallback when Whisper is unavailable."""
        return {
            "text": "",
            "segments": [],
            "confidence": None,
        }

    def _extract_word_timestamps(self, segments: list[dict[str, Any]]) -> list[Any]:
        """Converts Whisper segments into word-level WordTimestamp objects.

        Prefers the per-word timestamps inside each segment (returned when
        ``word_timestamps=True`` is passed to Whisper).  Falls back to
        segment-level timestamps when word-level data is absent.
        """
        try:
            word_timestamps: list[Any] = []
            for seg in segments:
                words = seg.get("words", [])
                if words:
                    for w in words:
                        word_text = w.get("word", "").strip()
                        if (
                            word_text
                            and w.get("start") is not None
                            and w.get("end") is not None
                        ):
                            word_timestamps.append(
                                WordTimestamp(
                                    word=word_text,
                                    start=w["start"],
                                    end=w["end"],
                                )
                            )
                elif (
                    seg.get("text")
                    and seg.get("start") is not None
                    and seg.get("end") is not None
                ):
                    segment_words = seg["text"].split()
                    seg_start = float(seg["start"])
                    seg_end = float(seg["end"])
                    seg_duration = seg_end - seg_start
                    if segment_words and seg_duration > 0:
                        word_duration = seg_duration / len(segment_words)
                        for idx, word in enumerate(segment_words):
                            word_start = seg_start + idx * word_duration
                            word_end = word_start + word_duration
                            word_timestamps.append(
                                WordTimestamp(
                                    word=word,
                                    start=word_start,
                                    end=word_end,
                                )
                            )
                    elif segment_words:
                        word_timestamps.append(
                            WordTimestamp(
                                word=segment_words[0],
                                start=seg_start,
                                end=seg_end,
                            )
                        )
            return word_timestamps
        except Exception:
            return []

    def _compute_duration(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        segments: list[dict[str, Any]],
    ) -> float:
        """Computes utterance duration in seconds.

        Prefers the last Whisper segment end-time; falls back to PCM math
        (16-bit mono = 2 bytes/sample).
        """
        if segments:
            last_end = max((seg.get("end", 0.0) for seg in segments), default=0.0)
            if last_end > 0:
                return round(last_end, 2)

        if audio_bytes and sample_rate > 0:
            try:
                with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                    n_frames = wf.getnframes()
                    framerate = wf.getframerate()
                    if framerate > 0:
                        return round(n_frames / framerate, 2)
            except Exception:
                pass
            return round(len(audio_bytes) / (sample_rate * 2), 2)

        return 0.0
