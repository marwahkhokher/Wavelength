"""Pipeline Orchestrator (Taha's ownership).

Sequential turn pipeline orchestrator connecting STT -> Tone -> Qwen Eval -> Prompt LLM -> ElevenLabs TTS.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-tech-infra" / "src"))

from wavelength_voice.ai_service.contracts import (
    PromptLLMInput,
    PromptLLMOutput,
    SessionContext,
    TurnRecord,
)

from ..qwen_evaluation.deep_evaluator import QwenDeepEvaluator
from ..stt_filler.stt_engine import WhisperSTTEngine
from ..tone_analysis.emotion_classifier import EmotionClassifier
from .conversation_llm import ConversationLLM
from .tts_stream import ElevenLabsTTSClient


class VoicePipelineOrchestrator:
    """Orchestrates the sequential turn pipeline across all AI team modules."""

    def __init__(self, session_context: SessionContext):
        self.session_context = session_context
        self.stt_engine = WhisperSTTEngine()
        self.emotion_classifier = EmotionClassifier()
        self.qwen_evaluator = QwenDeepEvaluator()
        self.prompt_llm = ConversationLLM()
        self.tts_client = ElevenLabsTTSClient()
        self.conversation_history: list[TurnRecord] = []
        self.turn_counter = 0

    async def process_user_turn(self, audio_bytes: bytes) -> PromptLLMOutput:
        """Executes the full sequential pipeline for one user turn:
        Audio -> STT + Tone -> Qwen Deep Evaluation -> Prompt LLM -> ElevenLabs TTS.
        """
        self.turn_counter += 1
        
        # Step 1: STT & Filler Detection (Areej)
        stt_result = await self.stt_engine.transcribe_audio_bytes(audio_bytes)
        
        # Step 2: Acoustic Tone & Emotion Analysis (Zaid)
        tone_result = self.emotion_classifier.classify_audio_tone(
            audio_bytes,
            word_count=stt_result.total_words,
            duration_sec=stt_result.utterance_duration_sec,
        )
        
        # Step 3: Qwen Deep Evaluation (Ahmed)
        qwen_eval = await self.qwen_evaluator.evaluate_turn(
            turn_index=self.turn_counter,
            stt_result=stt_result,
            tone_result=tone_result,
            session_context=self.session_context,
        )
        
        # Step 4: Prompt Generation LLM (Taha)
        llm_input = PromptLLMInput(
            session_context=self.session_context,
            current_stt=stt_result,
            current_tone=tone_result,
            conversation_history=self.conversation_history,
        )
        llm_output = await self.prompt_llm.generate_next_turn(llm_input, qwen_eval)
        
        # Record Turn History
        self.conversation_history.append(
            TurnRecord(
                turn_index=self.turn_counter,
                user_transcript=stt_result.transcript,
                user_tone=tone_result.primary_emotion,
                ai_response=llm_output.reply_text,
                eval_scores=qwen_eval.scores,
            )
        )
        
        return llm_output
