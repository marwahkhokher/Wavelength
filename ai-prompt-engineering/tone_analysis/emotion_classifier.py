"""emotion2vec / wav2vec2 Emotion Classifier (Zaid's ownership).

Extracts primary emotion classification and produces ToneResult.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-tech-infra" / "src"))

from wavelength_voice.ai_service.contracts import PauseMetrics, PrimaryEmotion, ToneResult
from .acoustic_metrics import calculate_acoustic_metrics


class EmotionClassifier:
    """emotion2vec / wav2vec2 model wrapper for acoustic emotion detection."""

    def __init__(self, model_name: str = "iic/emotion2vec_plus_large"):
        self.model_name = model_name

    def classify_audio_tone(
        self, audio_bytes: bytes, word_count: int, duration_sec: float
    ) -> ToneResult:
        """Classifies audio emotion state and combines with Python acoustic metrics."""
        # Simulated emotion classification output
        scores = {
            "hesitant": 0.75,
            "anxious": 0.15,
            "confident": 0.10,
        }
        primary: PrimaryEmotion = "hesitant"
        
        metrics = calculate_acoustic_metrics(word_count, duration_sec)
        
        return ToneResult(
            primary_emotion=primary,
            emotion_confidence_scores=scores,
            pitch_energy_variation=0.45,
            hesitation_score=0.65,
            pause_metrics=metrics["pause_metrics"],  # type: ignore
            speech_rate_wpm=metrics["speech_rate_wpm"],  # type: ignore
            silence_ratio=metrics["silence_ratio"],  # type: ignore
        )
