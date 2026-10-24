"""Environment-backed application configuration."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings with safe local defaults and explicit provider configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Applied Systems AI Assistant"
    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:8501"
    app_api_key: SecretStr | None = None

    agent_mode: Literal["offline", "live"] = "offline"
    agent_model: str = "gpt-4o-mini"
    agent_base_url: str = "https://api.openai.com/v1"
    agent_api_key: SecretStr | None = None
    agent_max_steps: int = Field(default=8, ge=1, le=20)

    primary_provider: str = "openai"
    primary_model: str = "gpt-4.1-mini"
    primary_base_url: str = "https://api.openai.com/v1"
    primary_api_key: SecretStr | None = None

    fallback_provider: str | None = None
    fallback_model: str | None = None
    fallback_base_url: str | None = None
    fallback_api_key: SecretStr | None = None

    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    max_tokens: int = Field(default=700, ge=64, le=8192)
    provider_timeout_seconds: float = Field(default=20.0, gt=0.0, le=300.0)
    provider_total_timeout_seconds: float = Field(default=60.0, gt=0.0, le=300.0)
    generation_timeout_seconds: float = Field(default=80.0, gt=0.0, le=300.0)
    retry_attempts: int = Field(default=2, ge=1, le=8)
    retry_base_delay_seconds: float = Field(default=0.25, ge=0.0, le=10.0)

    embedding_provider: Literal["fastembed", "hash"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = Field(default=384, ge=8, le=4096)
    vector_store: Literal["qdrant", "memory"] = "qdrant"
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "assistant_knowledge"
    retrieval_top_k: int = Field(default=4, ge=1, le=20)
    minimum_relevance_score: float = Field(default=0.20, ge=-1.0, le=1.0)
    chunk_size: int = Field(default=900, ge=200, le=8000)
    chunk_overlap: int = Field(default=140, ge=0, le=2000)
    max_context_characters: int = Field(default=8000, ge=500, le=50_000)
    max_prompt_characters: int = Field(default=12_000, ge=2000, le=200_000)
    max_prompt_utf8_bytes: int = Field(default=18_000, ge=2000, le=500_000)
    max_upload_bytes: int = Field(default=10_000_000, ge=1024, le=100_000_000)
    max_document_characters: int = Field(default=2_000_000, ge=1000, le=20_000_000)
    max_document_chunks: int = Field(default=2500, ge=1, le=20_000)
    max_pdf_pages: int = Field(default=500, ge=1, le=10_000)
    max_request_body_bytes: int = Field(default=5_000_000, ge=1024, le=100_000_000)
    embedding_batch_size: int = Field(default=64, ge=1, le=512)
    embedding_concurrency: int = Field(default=2, ge=1, le=16)
    retrieval_concurrency: int = Field(default=4, ge=1, le=32)
    ingestion_concurrency: int = Field(default=2, ge=1, le=16)

    redis_url: str | None = None
    redis_timeout_seconds: float = Field(default=2.0, gt=0.0, le=30.0)
    cache_ttl_seconds: int = Field(default=300, ge=0, le=86_400)
    rate_limit_requests: int = Field(default=30, ge=1, le=100_000)
    rate_limit_window_seconds: int = Field(default=60, ge=1, le=86_400)
    batch_concurrency: int = Field(default=4, ge=1, le=32)

    @field_validator(
        "fallback_provider",
        "fallback_model",
        "fallback_base_url",
        "fallback_api_key",
        "app_api_key",
        "redis_url",
        mode="before",
    )
    @classmethod
    def empty_string_is_unset(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("collection_name")
    @classmethod
    def validate_collection_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("collection_name must not be empty")
        return normalized

    @model_validator(mode="after")
    def validate_chunk_window(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    @computed_field
    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one validated settings object per process."""

    return Settings()
