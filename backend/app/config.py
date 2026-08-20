from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    database_url: str

    # Redis
    redis_url: str = "redis://redis:6379/0"

    # Auth
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 14

    # LLM
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"

    # Storage
    storage_backend: str = "local"
    s3_bucket: str = ""
    s3_region: str = ""

    # Voice provider
    voice_provider: str = "openai_realtime"
    voice_provider_api_key: str = ""


settings = Settings()
