"""Tone & Emotion Analysis module (Zaid's ownership)."""

from .emotion_classifier import EmotionClassifier
from .acoustic_metrics import calculate_acoustic_metrics

__all__ = ["EmotionClassifier", "calculate_acoustic_metrics"]
