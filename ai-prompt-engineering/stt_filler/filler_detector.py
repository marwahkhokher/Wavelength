"""Filler Word Detector (Areej's ownership).

Detects timestamped filler words ('um', 'uh', 'like', 'you know', 'basically', 'actually') from transcript.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-tech-infra" / "src"))

from wavelength_voice.ai_service.contracts import FillerWordOccurrence, WordTimestamp

DEFAULT_FILLER_WORDS = {"um", "uh", "like", "you know", "basically", "actually", "so"}


class FillerDetector:
    """Detects filler words from transcripts and timestamped word arrays."""

    def __init__(self, filler_lexicon: set[str] | None = None):
        self.filler_lexicon = filler_lexicon or DEFAULT_FILLER_WORDS

    def detect_fillers(
        self, transcript: str, word_timestamps: list[WordTimestamp] | None = None
    ) -> list[FillerWordOccurrence]:
        """Scans transcript and returns detected filler occurrences."""
        occurrences: list[FillerWordOccurrence] = []
        
        if word_timestamps:
            for item in word_timestamps:
                clean_word = re.sub(r"[^\w]", "", item.word.lower())
                if clean_word in self.filler_lexicon:
                    occurrences.append(
                        FillerWordOccurrence(
                            word=clean_word,
                            start_time=item.start,
                            end_time=item.end,
                        )
                    )
        else:
            # Basic fallback when word timestamps are not provided
            words = transcript.lower().split()
            current_time = 0.0
            for w in words:
                clean_word = re.sub(r"[^\w]", "", w)
                if clean_word in self.filler_lexicon:
                    occurrences.append(
                        FillerWordOccurrence(
                            word=clean_word,
                            start_time=current_time,
                            end_time=current_time + 0.3,
                        )
                    )
                current_time += 0.4
                
        return occurrences
