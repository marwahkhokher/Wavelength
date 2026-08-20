"""Python Acoustic Metric Calculator (Zaid's ownership).

Calculates deterministic quantitative metrics: WPM, silence ratio, speech pauses, pitch variation.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
