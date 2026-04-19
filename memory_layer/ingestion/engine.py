from __future__ import annotations

import hashlib
import importlib
import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

from memory_layer.exceptions import IngestionError, MemoryLayerError
from memory_layer.ingestion.chunker import SemanticChunker
from memory_layer.ingestion.embedder import SentenceTransformerEmbedder
from memory_layer.ingestion.scorer import ImportanceScorer
from memory_layer.models import MemoryChunk, MemoryType
from memory_layer.plugins import MemoryTypePluginRegistry, get_default_plugin_registry
from memory_layer.storage.base import (
    MemorySearchQuery,
    ProceduralMemoryRecord,
    SessionStatsRecord,
    StorageBackend,
)


def _utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(UTC)


def _whitespace_token_count(text: str) -> int:
    """Return a fallback token estimate based on whitespace splitting."""
    return len(text.split())


def _build_tiktoken_counter(encoding_name: str) -> Callable[[str], int] | None:
    """Build a tiktoken-based token counter when dependency is available."""
    try:
        tiktoken = importlib.import_module("tiktoken")
    except ModuleNotFoundError:
        return None

    encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(text: str) -> int:
        return len(encoding.encode(text))

    return count_tokens


class Chunker(Protocol):
    """Protocol for ingestion chunking components."""

    def chunk(self, text: str) -> list[str]:
        """Split source text into semantic chunks."""


class Embedder(Protocol):
    """Protocol for ingestion embedding components."""

    async def encode_batch(self, chunks: list[str]) -> list[list[float]]:
        """Encode chunks into vector embeddings."""


class Scorer(Protocol):
    """Protocol for ingestion importance scoring components."""

    threshold: float

    def score(
        self,
        chunk_text: str,
        *,
        chunk_embedding: list[float],
        existing_embeddings: list[list[float]] | None = None,
    ) -> float:
        """Return bounded importance score for one chunk."""


class IngestionEngine:
    """Ingestion orchestrator for chunking, embedding, scoring, and persistence."""

    _PREFERENCE_SIGNAL_RE = re.compile(
        r"\b(i\s+prefer|i\s+always|i\s+hate|my\s+favorite)\b",
        re.IGNORECASE,
    )
    _WORKING_SIGNAL_RE = re.compile(
        r"\b(now|right\s+now|currently|this\s+session|today)\b",
        re.IGNORECASE,
    )
    _SEMANTIC_FACT_RE = re.compile(
        r"\b(i|user|[A-Z][a-z]+)\s+(am|is|are|has|have|work(?:s)?|use(?:s)?)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        *,
        storage: StorageBackend,
        chunker: Chunker | None = None,
        embedder: Embedder | None = None,
        scorer: Scorer | None = None,
        novelty_top_k: int = 20,
        embedding_history_limit: int = 256,
        token_counter: Callable[[str], int] | None = None,
        encoding_name: str = "cl100k_base",
        now_provider: Callable[[], datetime] | None = None,
        plugin_registry: MemoryTypePluginRegistry | None = None,
    ) -> None:
        if novelty_top_k <= 0:
            raise ValueError("novelty_top_k must be greater than zero")
        if embedding_history_limit <= 0:
            raise ValueError("embedding_history_limit must be greater than zero")

        self._storage = storage
        self._chunker = chunker or SemanticChunker()
        self._embedder = embedder or SentenceTransformerEmbedder()
        self._scorer = scorer or ImportanceScorer()
        self._novelty_top_k = novelty_top_k
        self._embedding_history_limit = embedding_history_limit
        self._token_count = (
            token_counter
            or _build_tiktoken_counter(encoding_name)
            or _whitespace_token_count
        )
        self._now = now_provider or _utc_now
        self._embedding_history: dict[str, list[list[float]]] = {}
        self._plugin_registry = plugin_registry or get_default_plugin_registry()

    async def ingest(
        self,
        text: str,
        user_id: str,
        session_id: str,
        memory_type_hint: MemoryType | None = None,
    ) -> list[MemoryChunk]:
        """Ingest text into persisted memory chunks for one user/session."""
        if not user_id:
            raise ValueError("user_id is required")
        if not session_id:
            raise ValueError("session_id is required")

        if not text.strip():
            return []

        try:
            chunk_texts = self._chunker.chunk(text)
            if not chunk_texts:
                return []

            embeddings = await self._embedder.encode_batch(chunk_texts)
            if len(embeddings) != len(chunk_texts):
                raise IngestionError("Embedding count does not match chunk count")

            existing_embeddings = await self._load_existing_embeddings(
                user_id=user_id,
                seed_embedding=embeddings[0],
            )

            accepted_chunks: list[MemoryChunk] = []
            for chunk_text, embedding in zip(chunk_texts, embeddings, strict=True):
                importance = self._scorer.score(
                    chunk_text,
                    chunk_embedding=embedding,
                    existing_embeddings=existing_embeddings,
                )
                if importance < self._scorer.threshold:
                    continue

                memory_type, metadata = self._resolve_memory_type(
                    chunk_text,
                    memory_type_hint=memory_type_hint,
                )
                timestamp = self._now()
                chunk = MemoryChunk(
                    id=self._memory_id(),
                    user_id=user_id,
                    session_id=session_id,
                    memory_type=memory_type,
                    content=chunk_text,
                    importance=importance,
                    token_count=self._token_count_for_chunk(chunk_text),
                    embedding=embedding,
                    created_at=timestamp,
                    updated_at=timestamp,
                    metadata=metadata,
                )
                accepted_chunks.append(chunk)
                existing_embeddings.append(embedding)

            if not accepted_chunks:
                return []

            persisted_chunks = await self._storage.upsert_memory_chunks(accepted_chunks)
            await self._persist_procedural_records(persisted_chunks)
            await self._update_session_stats(
                user_id=user_id,
                session_id=session_id,
                persisted_chunks=persisted_chunks,
            )
            self._append_embedding_history(user_id=user_id, chunks=accepted_chunks)
            return persisted_chunks
        except MemoryLayerError as exc:
            if isinstance(exc, IngestionError):
                raise
            raise IngestionError(f"Failed to ingest text: {exc}") from exc
        except Exception as exc:
            raise IngestionError(f"Failed to ingest text: {exc}") from exc

    async def _load_existing_embeddings(
        self,
        *,
        user_id: str,
        seed_embedding: list[float],
    ) -> list[list[float]]:
        """Load candidate embeddings for novelty estimation."""
        cached = [vector[:] for vector in self._embedding_history.get(user_id, [])]
        if not seed_embedding:
            return cached

        try:
            candidates = await self._storage.query_vectors(
                MemorySearchQuery(
                    user_id=user_id,
                    query_embedding=seed_embedding,
                    top_k=self._novelty_top_k,
                    include_compressed=True,
                    min_importance=0.0,
                )
            )
        except Exception:
            return cached

        fetched = [
            vector[:] for chunk in candidates if (vector := chunk.embedding) is not None
        ]
        return [*cached, *fetched]

    async def _persist_procedural_records(
        self,
        chunks: list[MemoryChunk],
    ) -> None:
        """Persist procedural chunks into key-value procedural memory storage."""
        for chunk in chunks:
            if chunk.memory_type is not MemoryType.PROCEDURAL:
                continue

            record = ProceduralMemoryRecord(
                user_id=chunk.user_id,
                key=self._procedural_key(chunk.content),
                value=chunk.content,
                confidence=chunk.importance,
                updated_at=self._now(),
                source_chunk_id=chunk.id,
            )
            await self._storage.upsert_procedural_memory(record)

    async def _update_session_stats(
        self,
        *,
        user_id: str,
        session_id: str,
        persisted_chunks: list[MemoryChunk],
    ) -> None:
        """Update aggregate per-session counters after successful persistence."""
        new_memory_count = len(persisted_chunks)
        new_total_tokens = sum(chunk.token_count for chunk in persisted_chunks)
        now = self._now()

        existing = await self._storage.get_session_stats(session_id=session_id)
        if existing is not None and existing.user_id != user_id:
            # Preserve isolation: do not mutate another user's session counters.
            return

        if existing is None:
            record = SessionStatsRecord(
                session_id=session_id,
                user_id=user_id,
                memory_count=new_memory_count,
                total_tokens_stored=new_total_tokens,
                started_at=now,
                last_activity=now,
            )
        else:
            record = SessionStatsRecord(
                session_id=session_id,
                user_id=user_id,
                memory_count=existing.memory_count + new_memory_count,
                total_tokens_stored=existing.total_tokens_stored + new_total_tokens,
                started_at=existing.started_at,
                last_activity=now,
                ended_at=existing.ended_at,
                compressed=existing.compressed,
            )

        await self._storage.upsert_session_stats(record)

    def _append_embedding_history(self, *, user_id: str, chunks: list[MemoryChunk]) -> None:
        """Keep a bounded in-memory history of recent user embeddings."""
        embeddings = [vector[:] for chunk in chunks if (vector := chunk.embedding) is not None]
        if not embeddings:
            return

        history = self._embedding_history.setdefault(user_id, [])
        history.extend(embeddings)
        if len(history) > self._embedding_history_limit:
            self._embedding_history[user_id] = history[-self._embedding_history_limit :]

    def _resolve_memory_type(
        self,
        chunk_text: str,
        *,
        memory_type_hint: MemoryType | None,
    ) -> tuple[MemoryType, dict[str, Any]]:
        """Resolve base memory type plus optional plugin metadata."""
        if memory_type_hint is not None:
            return memory_type_hint, {}

        plugin = self._plugin_registry.match(chunk_text)
        if plugin is None:
            return self._classify_memory_type(chunk_text), {}

        metadata: dict[str, Any] = {"custom_memory_type": plugin.name}
        plugin_metadata = plugin.metadata(chunk_text)
        if plugin_metadata:
            for key, value in plugin_metadata.items():
                if key == "custom_memory_type":
                    continue
                metadata[key] = value
        return plugin.base_memory_type, metadata

    @classmethod
    def _classify_memory_type(cls, chunk_text: str) -> MemoryType:
        """Route chunk to one of the supported memory types."""
        if cls._PREFERENCE_SIGNAL_RE.search(chunk_text):
            return MemoryType.PROCEDURAL
        if cls._WORKING_SIGNAL_RE.search(chunk_text):
            return MemoryType.WORKING
        if cls._SEMANTIC_FACT_RE.search(chunk_text):
            return MemoryType.SEMANTIC
        return MemoryType.EPISODIC

    @staticmethod
    def _memory_id() -> str:
        """Generate canonical memory identifier."""
        return f"mem_{uuid4().hex[:12]}"

    def _token_count_for_chunk(self, chunk_text: str) -> int:
        """Return token count with a minimum of one token for non-empty chunks."""
        token_count = self._token_count(chunk_text)
        return token_count if token_count > 0 else 1

    @staticmethod
    def _procedural_key(chunk_text: str) -> str:
        """Generate stable procedural key from normalized preference text."""
        normalized = re.sub(r"\s+", " ", chunk_text.strip().lower())
        digest = hashlib.sha256(normalized.encode()).hexdigest()[:12]
        return f"preference_{digest}"


__all__ = ["IngestionEngine"]
