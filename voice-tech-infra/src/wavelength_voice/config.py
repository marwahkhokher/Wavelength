"""Runtime configuration, sourced from environment variables / .env."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    # --- Whisper Small (batch STT, replaces Deepgram) ---
    whisper_model_size: str = "small"
    whisper_language: str = "auto"
    whisper_sample_rate: int = 16000

    # --- ElevenLabs (streaming TTS) ---
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # default demo voice
    elevenlabs_model_id: str = "eleven_turbo_v2_5"
    elevenlabs_output_format: str = "pcm_16000"

    # --- AI service (owned by AI/Prompt Engineering team) ---
    ai_service_base_url: str = "http://localhost:9000"
    ai_service_timeout_seconds: float = 10.0
    use_mock_ai_service: bool = True

    # --- Session state ---
    session_reconnect_grace_seconds: float = 60.0
    session_expiry_sweep_interval_seconds: float = 15.0

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"


@lru_cache
def get_settings() -> Settings:
    return Settings()
