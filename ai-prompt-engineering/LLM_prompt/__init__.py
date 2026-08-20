"""LLM Prompt & Real-Time Voice Orchestration Package (Taha's ownership).

Provides the complete Person 5 component:
  1. Main Conversation LLM (Gemini 2.0 / Mock fallback)
  2. Modular Prompt Builder & Persona Engine
  3. Conversation Memory & Structured Context Tracking
  4. Real-Time Conversation Orchestrator combining Person 1-4 outputs
  5. ElevenLabs Streaming TTS with Instant Barge-In Interruption Handler
  6. AI Service HTTP Wire Server (FastAPI @ localhost:9000 for voice-tech-infra)
"""

from prompt_orchestration import (
    AIServiceSettings,
    BaseTTSClient,
    ConversationMemory,
    ConversationMessage,
    ConversationOrchestrator,
    ConversationState,
    DIFFICULTY_RULES,
    ElevenLabsTTSClient,
    GeminiLLMClient,
    InterviewerLLM,
    MockLLMClient,
    MockTTSClient,
    PersonaEngine,
    PersonaState,
    PromptBuilder,
    SessionManager,
    SessionState,
    StructuredMemory,
    VoicePipelineOrchestrator,
    ai_service_app,
    get_ai_settings,
    get_llm_client,
)

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
