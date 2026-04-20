from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from memory_vault.models import MemoryChunk, MemoryType, PaginatedResult


def _utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class MemorySearchQuery:
    """Vector-search request payload for user-scoped memory retrieval."""

    user_id: str
    query_embedding: Sequence[float]
    top_k: int = 20
    memory_types: tuple[MemoryType, ...] | None = None
    include_compressed: bool = False
    min_importance: float = 0.0

    def __post_init__(self) -> None:
        if not self.user_id:
            raise ValueError("user_id is required")
        if not self.query_embedding:
            raise ValueError("query_embedding cannot be empty")
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not 0.0 <= self.min_importance <= 1.0:
            raise ValueError("min_importance must be between 0.0 and 1.0")


@dataclass(frozen=True, slots=True)
class MemoryListQuery:
    """Metadata list request payload for paginated memory browsing."""

    user_id: str
    memory_type: MemoryType | None = None
    include_compressed: bool = False
    page: int = 1
    page_size: int = 20

    def __post_init__(self) -> None:
        if not self.user_id:
            raise ValueError("user_id is required")
        if self.page <= 0:
            raise ValueError("page must be greater than zero")
        if self.page_size <= 0:
            raise ValueError("page_size must be greater than zero")


@dataclass(frozen=True, slots=True)
class ProceduralMemoryRecord:
    """Represents a user preference entry stored in metadata backend."""

    user_id: str
    key: str
    value: str
    confidence: float = 1.0
    updated_at: datetime = field(default_factory=_utc_now)
    source_chunk_id: str | None = None

    def __post_init__(self) -> None:
        if not self.user_id:
            raise ValueError("user_id is required")
        if not self.key:
            raise ValueError("key is required")
        if not self.value:
            raise ValueError("value is required")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")


@dataclass(frozen=True, slots=True)
class SessionStatsRecord:
    """Session metadata surfaced by session statistics endpoints."""

    session_id: str
    user_id: str
    memory_count: int = 0
    total_tokens_stored: int = 0
    started_at: datetime = field(default_factory=_utc_now)
    last_activity: datetime = field(default_factory=_utc_now)
    ended_at: datetime | None = None
    compressed: bool = False

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id is required")
        if not self.user_id:
            raise ValueError("user_id is required")
        if self.memory_count < 0:
            raise ValueError("memory_count cannot be negative")
        if self.total_tokens_stored < 0:
            raise ValueError("total_tokens_stored cannot be negative")
        if self.last_activity < self.started_at:
            raise ValueError("last_activity must be greater than or equal to started_at")


class BackendLifecycle(ABC):
    """Common lifecycle contract shared by all storage backends."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize backend resources."""

    @abstractmethod
    async def close(self) -> None:
        """Close backend resources."""

    @abstractmethod
    async def healthcheck(self) -> dict[str, str]:
        """Return a backend-specific health payload."""

    async def __aenter__(self) -> BackendLifecycle:
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.close()


class VectorStoreBackend(BackendLifecycle, ABC):
    """Vector-store contract for ANN search and vector lifecycle operations."""

    @abstractmethod
    async def upsert_vectors(self, chunks: Sequence[MemoryChunk]) -> None:
        """Persist or update vector entries for chunks that include embeddings."""

    @abstractmethod
    async def query_vectors(self, query: MemorySearchQuery) -> list[MemoryChunk]:
        """Return candidate chunks scored for similarity to the query embedding."""

    @abstractmethod
    async def delete_vectors(self, memory_ids: Sequence[str], *, user_id: str) -> int:
        """Delete vectors for the provided memory IDs and user."""

    @abstractmethod
    async def delete_vectors_for_user(self, *, user_id: str) -> int:
        """Delete all vectors associated with a user."""


class MetadataStoreBackend(BackendLifecycle, ABC):
    """Metadata-store contract for chunk, procedural, and session state."""

    @abstractmethod
    async def upsert_memory_chunks(self, chunks: Sequence[MemoryChunk]) -> list[MemoryChunk]:
        """Persist or update memory chunk metadata records."""

    @abstractmethod
    async def get_memory_chunk(self, *, memory_id: str, user_id: str) -> MemoryChunk | None:
        """Fetch a single memory chunk by ID and user scope."""

    @abstractmethod
    async def list_memory_chunks(self, query: MemoryListQuery) -> PaginatedResult[MemoryChunk]:
        """Return paginated memory chunks for a user."""

    @abstractmethod
    async def delete_memory_chunk(self, *, memory_id: str, user_id: str) -> bool:
        """Delete one memory chunk record by ID and user."""

    @abstractmethod
    async def delete_memory_chunks_for_user(self, *, user_id: str) -> int:
        """Delete all memory chunk records for a user."""

    @abstractmethod
    async def upsert_procedural_memory(
        self,
        record: ProceduralMemoryRecord,
    ) -> ProceduralMemoryRecord:
        """Persist or update a procedural memory key-value entry."""

    @abstractmethod
    async def list_procedural_memory(self, *, user_id: str) -> list[ProceduralMemoryRecord]:
        """Return all procedural memory entries for a user."""

    @abstractmethod
    async def delete_procedural_memory(self, *, user_id: str, key: str) -> bool:
        """Delete a procedural memory entry by key for a user."""

    @abstractmethod
    async def upsert_session_stats(self, record: SessionStatsRecord) -> SessionStatsRecord:
        """Persist or update session statistics."""

    @abstractmethod
    async def get_session_stats(self, *, session_id: str) -> SessionStatsRecord | None:
        """Fetch session statistics by session ID."""


class StorageBackend(VectorStoreBackend, MetadataStoreBackend, ABC):
    """Unified storage abstraction used by core feature code."""


__all__ = [
    "BackendLifecycle",
    "MemoryListQuery",
    "MemorySearchQuery",
    "MetadataStoreBackend",
    "ProceduralMemoryRecord",
    "SessionStatsRecord",
    "StorageBackend",
    "VectorStoreBackend",
]
