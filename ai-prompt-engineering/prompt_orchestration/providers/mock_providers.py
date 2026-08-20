"""Mock provider implementations (Taha's ownership).

Used when teammates' real implementations are unavailable. Return realistic
data matching the shared contract schemas so the orchestrator can be developed
and tested independently.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "voice-tech-infra" / "src"))

from wavelength_voice.ai_service.contracts import (
    BaselineMetrics,
    FillerWordOccurrence,
    MetricScores,
    PauseMetrics,
    PerTurnEvaluation,
    PersonaProfile,
    ScenarioConfig,
    SessionContext,
    STTResult,
    ToneResult,
)

from .base import AnalysisProvider, SessionContextProvider, ToneProvider, TranscriptProvider


class MockTranscriptProvider(TranscriptProvider):
    """Returns simulated transcript data matching STTResult schema."""

    _SAMPLE_TRANSCRIPTS = [
        "I believe I deserve this opportunity because I've consistently exceeded my targets.",
        "Um, well, I think my contribution to the project, like, increased team velocity significantly.",
        "Over the past year I led three major initiatives that resulted in a 30 percent revenue increase.",
        "Honestly, I'm not sure how to answer that, but I think I've been doing good work.",
        "I managed a team of four developers and we delivered the project two weeks ahead of schedule.",
    ]

    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> STTResult:
        t0 = time.perf_counter()
        transcript = random.choice(self._SAMPLE_TRANSCRIPTS)
        words = transcript.split()
        duration_sec = max(1.0, len(audio_bytes) / (sample_rate * 2))

        # Detect simple fillers
        filler_words_set = {"um", "uh", "like", "well", "basically", "actually", "so"}
        fillers: list[FillerWordOccurrence] = []
        current_time = 0.0
        for w in words:
            clean = w.strip(".,!?").lower()
            if clean in filler_words_set:
                fillers.append(
                    FillerWordOccurrence(word=clean, start_time=current_time, end_time=current_time + 0.3)
                )
            current_time += 0.35

        latency_ms = (time.perf_counter() - t0) * 1000

        return STTResult(
            transcript=transcript,
            is_final=True,
            total_words=len(words),
            filler_word_count=len(fillers),
            filler_words=fillers,
            utterance_duration_sec=duration_sec,
            stt_latency_ms=latency_ms,
        )


class MockToneProvider(ToneProvider):
    """Returns simulated tone analysis matching ToneResult schema."""

    async def analyze_tone(
        self,
        audio_bytes: bytes,
        word_count: int,
        duration_sec: float,
    ) -> ToneResult:
        emotions = {
            "neutral": round(random.uniform(0.3, 0.7), 2),
            "confident": round(random.uniform(0.05, 0.3), 2),
            "hesitant": round(random.uniform(0.05, 0.4), 2),
            "anxious": round(random.uniform(0.02, 0.15), 2),
        }
        primary = max(emotions, key=emotions.get)  # type: ignore[arg-type]

        wpm = round((word_count / max(1.0, duration_sec)) * 60, 1) if duration_sec > 0 else 0.0

        return ToneResult(
            primary_emotion=primary,  # type: ignore[arg-type]
            emotion_confidence_scores=emotions,
            pitch_energy_variation=round(random.uniform(0.2, 0.6), 2),
            hesitation_score=round(random.uniform(0.1, 0.7), 2),
            pause_metrics=PauseMetrics(
                total_pause_duration_sec=round(random.uniform(0.3, 1.5), 2),
                pause_count=random.randint(1, 4),
                max_pause_sec=round(random.uniform(0.2, 0.8), 2),
            ),
            speech_rate_wpm=wpm,
            silence_ratio=round(random.uniform(0.1, 0.3), 2),
        )


class MockAnalysisProvider(AnalysisProvider):
    """Returns simulated per-turn evaluation matching PerTurnEvaluation schema."""

    async def evaluate_turn(
        self,
        turn_index: int,
        stt_result: STTResult,
        tone_result: ToneResult,
    ) -> PerTurnEvaluation:
        # Generate realistic scores with some variance
        base = random.uniform(55, 85)
        scores = MetricScores(
            clarity=round(min(100, max(0, base + random.uniform(-10, 10))), 1),
            empathy=round(min(100, max(0, base + random.uniform(-15, 5))), 1),
            filler_words_score=round(min(100, max(0, 100 - stt_result.filler_word_count * 15)), 1),
            structure=round(min(100, max(0, base + random.uniform(-10, 5))), 1),
            relevance=round(min(100, max(0, base + random.uniform(-5, 15))), 1),
            confidence=round(min(100, max(0, base + random.uniform(-15, 10))), 1),
            overall_turn_score=round(base, 1),
        )

        strengths = []
        improvements = []

        if scores.relevance > 80:
            strengths.append("High relevance to the scenario topic")
        if scores.clarity > 80:
            strengths.append("Clear and well-articulated response")
        if scores.confidence < 60:
            improvements.append("Show more confidence — avoid hedging language")
        if stt_result.filler_word_count > 1:
            improvements.append("Reduce filler word usage for better fluency")
        if scores.structure < 65:
            improvements.append("Structure responses using the STAR method")

        if not strengths:
            strengths.append("Addressed the question directly")
        if not improvements:
            improvements.append("Continue maintaining this level of quality")

        return PerTurnEvaluation(
            turn_index=turn_index,
            scores=scores,
            strengths=strengths,
            areas_for_improvement=improvements,
            coach_tip="Pause for 1 second before speaking to organize your thoughts.",
        )


class MockSessionContextProvider(SessionContextProvider):
    """Returns a simulated SessionContext matching the schema."""

    def build_context(
        self,
        session_id: str,
        user_id: str,
        mode: str = "professional",
        scenario_title: str = "Salary Negotiation",
        scenario_description: str = "Negotiating a raise with your manager.",
        persona_name: str = "David Miller",
        persona_role: str = "VP of Engineering",
        difficulty: str = "medium",
        duration_seconds: int = 300,
    ) -> SessionContext:
        if mode == "professional":
            comm_style = (
                "Measured, professional, face-saving. Avoids blunt refusals. "
                "Uses diplomatic language and structured objections."
            )
            attitude = "firm but receptive"
            tone_traits = ["diplomatic", "structured", "face-saving"]
        else:
            comm_style = (
                "Direct, casual, conversational. Uses informal language, "
                "code-switching, and colloquial expressions."
            )
            attitude = "casual and candid"
            tone_traits = ["colloquial", "direct", "informal"]

        return SessionContext(
            session_id=session_id,
            user_id=user_id,
            mode=mode,  # type: ignore[arg-type]
            difficulty=difficulty,  # type: ignore[arg-type]
            duration_seconds=duration_seconds,
            scenario=ScenarioConfig(
                title=scenario_title,
                description=scenario_description,
                setting="Live Session Practice",
            ),
            persona=PersonaProfile(
                name=persona_name,
                role_description=persona_role,
                communication_style=comm_style,
                attitude=attitude,
                tone_traits=tone_traits,
            ),
            baseline_metrics=BaselineMetrics(),
        )
