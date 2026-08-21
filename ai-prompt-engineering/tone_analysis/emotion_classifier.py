"""emotion2vec / wav2vec2 Emotion Classifier (Zaid's ownership).

Extracts primary emotion classification and produces ToneResult.json.
"""
"""emotion2vec / wav2vec2 Emotion Classifier (Zaid's ownership).

Extracts primary emotion classification and produces ToneResult.json.

NOTE: Requires additional dependencies not yet in requirements.txt.
Run this before using this module:
    pip install librosa funasr torch torchaudio

TODO: confirm with team where these belong (root requirements.txt vs
a new ai-prompt-engineering/requirements.txt) and move this there.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from funasr import AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-tech-infra" / "src"))

from wavelength_voice.ai_service.contracts import PauseMetrics, PrimaryEmotion, ToneResult
from .acoustic_metrics import (
    calculate_acoustic_metrics,
    extract_pitch_energy_variation,
    extract_silence_intervals,
)

# Maps emotion2vec's raw output labels to our contract's allowed PrimaryEmotion values.
EMOTION2VEC_TO_CONTRACT: dict[str, PrimaryEmotion] = {
    "angry": "frustrated",
    "disgusted": "frustrated",
    "fearful": "anxious",
    "happy": "energetic",
    "neutral": "neutral",
    "other": "neutral",
    "sad": "hesitant",
    "surprised": "energetic",
    "<unk>": "neutral",
}


class EmotionClassifier:
    """emotion2vec / wav2vec2 model wrapper for acoustic emotion detection."""

    def __init__(self, model_name: str = "iic/emotion2vec_plus_large"):
        self.model_name = model_name
        self._model = AutoModel(model=model_name)

    def classify_audio_tone(
        self, audio_bytes: bytes, word_count: int, duration_sec: float
    ) -> ToneResult:
        """Classifies audio emotion state and combines with Python acoustic metrics."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            result = self._model.generate(tmp.name, granularity="utterance")

        raw_labels = result[0]["labels"]
        raw_scores = result[0]["scores"]

        # Labels come back as "中立/neutral" — take the English part after "/".
        clean_labels = [label.split("/")[-1] for label in raw_labels]

        scores: dict[str, float] = {}
        for label, score in zip(clean_labels, raw_scores):
            mapped = EMOTION2VEC_TO_CONTRACT.get(label, "neutral")
            scores[mapped] = round(scores.get(mapped, 0.0) + float(score), 4)

        primary: PrimaryEmotion = max(scores, key=scores.get)  # type: ignore

        silence_intervals = extract_silence_intervals(audio_bytes)
        metrics = calculate_acoustic_metrics(word_count, duration_sec, silence_intervals)
        pitch_energy_variation = extract_pitch_energy_variation(audio_bytes)

        pause_metrics = metrics["pause_metrics"]
        hesitation_score = round(
            min(1.0, (pause_metrics.pause_count * 0.1) + metrics["silence_ratio"]), 2
        )

        return ToneResult(
            primary_emotion=primary,
            emotion_confidence_scores=scores,
            pitch_energy_variation=pitch_energy_variation,
            hesitation_score=hesitation_score,
            pause_metrics=pause_metrics,  # type: ignore
            speech_rate_wpm=metrics["speech_rate_wpm"],  # type: ignore
            silence_ratio=metrics["silence_ratio"],  # type: ignore
        )