"""Python Acoustic Metric Calculator (Zaid's ownership).

Calculates deterministic quantitative metrics: WPM, silence ratio, speech pauses, pitch variation.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-tech-infra" / "src"))

from wavelength_voice.ai_service.contracts import PauseMetrics


def calculate_words_per_minute(word_count: int, duration_sec: float) -> float:
    """Calculates Words Per Minute (WPM) via Python math."""
    if duration_sec <= 0:
        return 0.0
    return round((word_count / duration_sec) * 60.0, 1)


def calculate_pause_metrics(silence_intervals: list[tuple[float, float]]) -> PauseMetrics:
    """Computes total pause duration, max pause length, and pause counts."""
    if not silence_intervals:
        return PauseMetrics(total_pause_duration_sec=0.0, pause_count=0, max_pause_sec=0.0)

    durations = [end - start for start, end in silence_intervals]
    return PauseMetrics(
        total_pause_duration_sec=round(sum(durations), 2),
        pause_count=len(durations),
        max_pause_sec=round(max(durations), 2) if durations else 0.0,
    )


def calculate_acoustic_metrics(
    word_count: int,
    duration_sec: float,
    silence_intervals: list[tuple[float, float]] | None = None,
) -> dict[str, float | PauseMetrics]:
    """Assembles Python acoustic calculation dict."""
    wpm = calculate_words_per_minute(word_count, duration_sec)
    pauses = calculate_pause_metrics(silence_intervals or [])
    silence_ratio = round(pauses.total_pause_duration_sec / max(1.0, duration_sec), 2)

    return {
        "speech_rate_wpm": wpm,
        "pause_metrics": pauses,
        "silence_ratio": silence_ratio,
    }


def extract_silence_intervals(
    audio_bytes: bytes, top_db: float = 30.0
) -> list[tuple[float, float]]:
    """Detects silence gaps in audio using an energy threshold (librosa)."""
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
    non_silent = librosa.effects.split(y, top_db=top_db)

    silence_intervals: list[tuple[float, float]] = []
    prev_end = 0.0
    for start_sample, end_sample in non_silent:
        start_sec = start_sample / sr
        if start_sec > prev_end:
            silence_intervals.append((prev_end, start_sec))
        prev_end = end_sample / sr

    total_duration = len(y) / sr
    if prev_end < total_duration:
        silence_intervals.append((prev_end, total_duration))

    return silence_intervals


def extract_pitch_energy_variation(audio_bytes: bytes) -> float:
    """Computes normalized pitch/energy variability (0-1) using RMS energy."""
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
    rms = librosa.feature.rms(y=y)[0]
    if rms.size == 0 or rms.mean() == 0:
        return 0.5
    variation = float(np.std(rms) / (np.mean(rms) + 1e-6))
    return round(min(1.0, variation), 2)