from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

T = TypeVar("T")


def _utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(UTC)


class MemoryType(StrEnum):
    """Supported memory categories used by ingestion and retrieval."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    WORKING = "working"
    PROCEDURAL = "procedural"


class MemoryConfig(BaseModel):
    """User-facing runtime config for retrieval and storage behavior."""

    token_budget: int = Field(default=2000, gt=0)
    top_k: int = Field(default=5, gt=0)
    compression_threshold: int = Field(default=10, gt=0)
    embedding_model: str = Field(default="all-MiniLM-L6-v2", min_length=1)
    storage_backend: Literal["chroma", "qdrant"] = "chroma"
    metadata_backend: Literal["sqlite", "postgres"] = "sqlite"

    chroma_path: str = "./data/chroma"
    sqlite_path: str = "./data/memory.db"

    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "memory_vault"
    postgres_url: str | None = None

    importance_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    reranker_enabled: bool = False
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        min_length=1,
    )
    max_chunk_tokens: int = Field(default=300, gt=0)
    min_chunk_tokens: int = Field(default=50, gt=0)

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @model_validator(mode="after")
    def validate_dependencies(self) -> MemoryConfig:
        """Validate dependent fields after model construction."""
        if self.min_chunk_tokens > self.max_chunk_tokens:
            raise ValueError("min_chunk_tokens must be less than or equal to max_chunk_tokens")
        if self.storage_backend == "qdrant" and not self.qdrant_url:
            raise ValueError("qdrant_url is required when storage_backend is 'qdrant'")
        if self.metadata_backend == "postgres" and not self.postgres_url:
            raise ValueError("postgres_url is required when metadata_backend is 'postgres'")
        return self


class MemoryChunk(BaseModel):
    """Canonical persisted memory record shared across modules."""

    id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    memory_type: MemoryType
    content: str = Field(min_length=1)

    importance: float = Field(ge=0.0, le=1.0)
    token_count: int = Field(ge=0)
    embedding: list[float] | None = None

    compressed: bool = False
    compression_source: bool = False
    source_session_id: str | None = None
    chroma_id: str | None = None

    relevance_score: float | None = Field(default=None, ge=0.0, le=1.0)

    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @model_validator(mode="after")
    def validate_timestamps(self) -> MemoryChunk:
        """Ensure timestamp ordering remains valid."""
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must be greater than or equal to created_at")
        if self.compression_source and not self.source_session_id:
            raise ValueError("source_session_id is required when compression_source is true")
        return self


class MemorySummary(BaseModel):
    """Compact memory shape used by ingestion responses."""

    id: str = Field(min_length=1)
    memory_type: MemoryType
    importance: float = Field(ge=0.0, le=1.0)
    token_count: int = Field(ge=0)
    created_at: datetime = Field(default_factory=_utc_now)

    model_config = ConfigDict(extra="forbid")


class SaveResult(BaseModel):
    """Result payload for memory save operations."""

    saved: list[MemorySummary] = Field(default_factory=list)
    discarded_count: int = Field(default=0, ge=0)

    model_config = ConfigDict(extra="forbid")


class RecallResult(BaseModel):
    """Result payload for recall operations."""

    memories: list[MemoryChunk] = Field(default_factory=list)
    total_tokens: int = Field(default=0, ge=0)
    budget_used: float = Field(default=0.0, ge=0.0, le=1.0)
    prompt_block: str = ""

    model_config = ConfigDict(extra="forbid")


class PaginatedResult(BaseModel, Generic[T]):
    """Generic pagination response container."""

    items: list[T] = Field(default_factory=list)
    total: int = Field(default=0, ge=0)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1)

    model_config = ConfigDict(extra="forbid")

    @property
    def total_pages(self) -> int:
        """Return total pages for the current page size."""
        if self.total == 0:
            return 0
        return ((self.total - 1) // self.page_size) + 1
