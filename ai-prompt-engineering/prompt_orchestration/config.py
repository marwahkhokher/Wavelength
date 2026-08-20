"""AI Service Configuration (Taha's ownership).

All secrets and tunables are sourced from environment variables / .env file.
Never hard-code API keys or commit .env files.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

# Automatically find and load .env from current directory, parent directory, or package directory
def _load_env_file() -> None:
    env_paths = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent / ".env",
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent.parent / "LLM_prompt" / ".env.example",
    ]
    
    # Try using python-dotenv first
    try:
        from dotenv import load_dotenv
        for p in env_paths:
            if p.is_file():
                load_dotenv(dotenv_path=str(p), override=False)
    except ImportError:
        # Simple manual fallback parser
        for p in env_paths:
            if p.is_file():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#") and "=" in line:
                                k, v = line.split("=", 1)
                                k, v = k.strip(), v.strip()
                                if k and k not in os.environ:
                                    os.environ[k] = v
                except Exception:
                    pass

_load_env_file()

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class AIServiceSettings(BaseSettings):
        model_config = SettingsConfigDict(
            env_file=".env",
            env_prefix="",
            extra="ignore",
        )

        # --- Gemini / Conversation LLM ---
        gemini_api_key: str = ""
        conversation_model: str = "gemini-2.0-flash"
        conversation_temperature: float = 0.7
        conversation_max_tokens: int = 512

        # --- ElevenLabs (for standalone AI-service TTS testing) ---
        elevenlabs_api_key: str = ""
        elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"
        elevenlabs_model_id: str = "eleven_turbo_v2_5"
        elevenlabs_output_format: str = "pcm_16000"

        # --- AI Service Server ---
        ai_service_host: str = "0.0.0.0"
        ai_service_port: int = 9000
        log_level: str = "info"

        # --- Provider Mode ---
        provider_mode: Literal["real", "mock"] = "mock"

        # --- Conversation Memory ---
        max_recent_turns: int = 6
        max_history_turns_before_summary: int = 10
        max_history_token_budget: int = 3000

        # --- Session ---
        default_difficulty: Literal["easy", "medium", "hard"] = "medium"
        default_duration_seconds: int = 300

except ImportError:
    from pydantic import BaseModel, Field

    class AIServiceSettings(BaseModel):
        # --- Gemini / Conversation LLM ---
        gemini_api_key: str = Field(default_factory=lambda: os.environ.get("GEMINI_API_KEY", ""))
        conversation_model: str = Field(default_factory=lambda: os.environ.get("CONVERSATION_MODEL", "gemini-2.0-flash"))
        conversation_temperature: float = Field(default_factory=lambda: float(os.environ.get("CONVERSATION_TEMPERATURE", "0.7")))
        conversation_max_tokens: int = Field(default_factory=lambda: int(os.environ.get("CONVERSATION_MAX_TOKENS", "512")))

        # --- ElevenLabs (for standalone AI-service TTS testing) ---
        elevenlabs_api_key: str = Field(default_factory=lambda: os.environ.get("ELEVENLABS_API_KEY", ""))
        elevenlabs_voice_id: str = Field(default_factory=lambda: os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"))
        elevenlabs_model_id: str = Field(default_factory=lambda: os.environ.get("ELEVENLABS_MODEL_ID", "eleven_turbo_v2_5"))
        elevenlabs_output_format: str = Field(default_factory=lambda: os.environ.get("ELEVENLABS_OUTPUT_FORMAT", "pcm_16000"))

        # --- AI Service Server ---
        ai_service_host: str = Field(default_factory=lambda: os.environ.get("AI_SERVICE_HOST", "0.0.0.0"))
        ai_service_port: int = Field(default_factory=lambda: int(os.environ.get("AI_SERVICE_PORT", "9000")))
        log_level: str = Field(default_factory=lambda: os.environ.get("LOG_LEVEL", "info"))

        # --- Provider Mode ---
        provider_mode: Literal["real", "mock"] = Field(default_factory=lambda: os.environ.get("PROVIDER_MODE", "mock"))  # type: ignore[assignment]

        # --- Conversation Memory ---
        max_recent_turns: int = Field(default_factory=lambda: int(os.environ.get("MAX_RECENT_TURNS", "6")))
        max_history_turns_before_summary: int = Field(default_factory=lambda: int(os.environ.get("MAX_HISTORY_TURNS_BEFORE_SUMMARY", "10")))
        max_history_token_budget: int = Field(default_factory=lambda: int(os.environ.get("MAX_HISTORY_TOKEN_BUDGET", "3000")))

        # --- Session ---
        default_difficulty: Literal["easy", "medium", "hard"] = Field(default_factory=lambda: os.environ.get("DEFAULT_DIFFICULTY", "medium"))  # type: ignore[assignment]
        default_duration_seconds: int = Field(default_factory=lambda: int(os.environ.get("DEFAULT_DURATION_SECONDS", "300")))


@lru_cache
def get_ai_settings() -> AIServiceSettings:
    return AIServiceSettings()
