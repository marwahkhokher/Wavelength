"""Forwarding module for LLM_prompt.tts_stream."""
from prompt_orchestration.tts_stream import (
    BaseTTSClient,
    ElevenLabsTTSClient,
    MockTTSClient,
)

__all__ = ["BaseTTSClient", "ElevenLabsTTSClient", "MockTTSClient"]
