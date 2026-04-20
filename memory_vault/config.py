from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from memory_vault.models import MemoryConfig


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    storage_backend: Literal["chroma", "qdrant"] = "chroma"
    metadata_backend: Literal["sqlite", "postgres"] = "sqlite"
    chroma_path: str = "./data/chroma"
    sqlite_path: str = "./data/memory.db"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "memory_vault"
    postgres_url: str | None = None

    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: Literal["cpu", "cuda", "mps"] = "cpu"
    embedding_batch_size: int = Field(default=32, gt=0)
    embedding_cache: bool = True

    default_token_budget: int = Field(default=2000, gt=0)
    default_top_k: int = Field(default=5, gt=0)
    importance_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    compression_threshold: int = Field(default=10, gt=0)
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_chunk_tokens: int = Field(default=300, gt=0)
    min_chunk_tokens: int = Field(default=50, gt=0)

    compression_model: str | None = None
    compression_api_key: str | None = None
    compression_api_base: str | None = None
    compression_sessions: int = Field(default=5, gt=0)

    api_key: str | None = None
    host: str = "0.0.0.0"
    port: int = Field(default=8000, gt=0)
    workers: int = Field(default=1, gt=0)
    cors_origins: str = "*"
    rate_limit_save: int = Field(default=100, gt=0)
    rate_limit_recall: int = Field(default=200, gt=0)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["text", "json"] = "text"
    log_sanitize: bool = False
    metrics_enabled: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ML_",
        case_sensitive=False,
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_dependencies(self) -> Settings:
        """Validate dependent settings based on selected backends."""
        if self.min_chunk_tokens > self.max_chunk_tokens:
            raise ValueError("min_chunk_tokens must be less than or equal to max_chunk_tokens")
        if self.storage_backend == "qdrant" and not self.qdrant_url:
            raise ValueError("qdrant_url is required when storage_backend is 'qdrant'")
        if self.metadata_backend == "postgres" and not self.postgres_url:
            raise ValueError("postgres_url is required when metadata_backend is 'postgres'")
        return self

    @property
    def cors_origins_list(self) -> list[str]:
        """Return normalized CORS origins."""
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    def to_memory_config(self) -> MemoryConfig:
        """Build a runtime memory config for SDK/core usage."""
        return MemoryConfig(
            token_budget=self.default_token_budget,
            top_k=self.default_top_k,
            compression_threshold=self.compression_threshold,
            embedding_model=self.embedding_model,
            storage_backend=self.storage_backend,
            metadata_backend=self.metadata_backend,
            chroma_path=self.chroma_path,
            sqlite_path=self.sqlite_path,
            qdrant_url=self.qdrant_url,
            qdrant_api_key=self.qdrant_api_key,
            qdrant_collection=self.qdrant_collection,
            postgres_url=self.postgres_url,
            importance_threshold=self.importance_threshold,
            reranker_enabled=self.reranker_enabled,
            reranker_model=self.reranker_model,
            max_chunk_tokens=self.max_chunk_tokens,
            min_chunk_tokens=self.min_chunk_tokens,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()
