"""Prompt LLM, ElevenLabs TTS & Orchestration module (Taha's ownership).

Exports:
  - ConversationOrchestrator: Central coordinator for the full multi-module pipeline
  - InterviewerLLM: Persona-consistent response generator
  - PromptBuilder: Modular prompt construction engine
  - PersonaEngine: Dynamic persona state & prompt manager
  - ConversationMemory: Three-tier context management (recent, structured, summary)
  - SessionManager: In-memory session state lifecycle
  - ElevenLabsTTSClient & MockTTSClient: Streaming TTS with instant barge-in cancellation
  - GeminiLLMClient & MockLLMClient: LLM clients
  - VoicePipelineOrchestrator: Backward-compatible sequential turn runner
"""

from .ai_service_app import app as ai_service_app
from .config import AIServiceSettings, get_ai_settings
from .conversation_memory import ConversationMemory
from .conversation_orchestrator import ConversationOrchestrator
from .interviewer_llm import InterviewerLLM
from .llm_client import GeminiLLMClient, MockLLMClient, get_llm_client
from .models import (
    ConversationMessage,
    ConversationState,
    PersonaState,
    SessionState,
    StructuredMemory,
)
from .persona_engine import DIFFICULTY_RULES, PersonaEngine
from .pipeline_orchestrator import VoicePipelineOrchestrator
from .prompt_builder import PromptBuilder
from .session_manager import SessionManager
from .tts_stream import BaseTTSClient, ElevenLabsTTSClient, MockTTSClient

__all__ = [
    "ConversationOrchestrator",
    "VoicePipelineOrchestrator",
    "InterviewerLLM",
    "PromptBuilder",
    "PersonaEngine",
    "ConversationMemory",
    "SessionManager",
    "ElevenLabsTTSClient",
    "MockTTSClient",
    "BaseTTSClient",
    "GeminiLLMClient",
    "MockLLMClient",
    "get_llm_client",
    "ConversationState",
    "PersonaState",
    "StructuredMemory",
    "ConversationMessage",
    "SessionState",
    "DIFFICULTY_RULES",
    "AIServiceSettings",
    "get_ai_settings",
    "ai_service_app",
]
