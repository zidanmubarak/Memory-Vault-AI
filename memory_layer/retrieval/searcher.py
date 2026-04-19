from __future__ import annotations

from collections.abc import Sequence

from memory_layer.exceptions import MemoryLayerError, RetrievalError
from memory_layer.models import MemoryChunk, MemoryType
from memory_layer.storage.base import MemorySearchQuery, StorageBackend


class MemorySearcher:
    """ANN retrieval entrypoint over storage vector search with candidate filtering."""

    def __init__(
        self,
        *,
        storage: StorageBackend,
        candidate_multiplier: int = 4,
        min_importance: float = 0.2,
    ) -> None:
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be greater than zero")
        if not 0.0 <= min_importance <= 1.0:
            raise ValueError("min_importance must be between 0.0 and 1.0")

        self._storage = storage
        self._candidate_multiplier = candidate_multiplier
        self._min_importance = min_importance

    async def search(
        self,
        *,
        user_id: str,
        query_embedding: Sequence[float],
        top_k: int = 5,
        memory_types: Sequence[MemoryType] | None = None,
        include_compressed: bool = False,
    ) -> list[MemoryChunk]:
        """Search and filter memory candidates for a user query embedding.

        Args:
            user_id: User identifier that scopes retrieval.
            query_embedding: Query vector embedding using the configured embedding model.
            top_k: Maximum number of filtered candidates to return.
            memory_types: Optional memory-type filter.
            include_compressed: Whether to include compressed memories when no summary exists.

        Returns:
            Filtered and relevance-sorted memory chunks.

        Raises:
            ValueError: If request inputs are invalid.
            RetrievalError: If storage query or filtering fails.
        """
        if not user_id:
            raise ValueError("user_id is required")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_vector = list(query_embedding)
        if not query_vector:
            raise ValueError("query_embedding cannot be empty")

        normalized_types = tuple(memory_types) if memory_types else None
        top_k_candidates = max(top_k, top_k * self._candidate_multiplier)

        try:
            candidates = await self._storage.query_vectors(
                MemorySearchQuery(
                    user_id=user_id,
                    query_embedding=query_vector,
                    top_k=top_k_candidates,
                    memory_types=normalized_types,
                    include_compressed=True,
                    min_importance=0.0,
                )
            )
            filtered = self._filter_candidates(
                candidates,
                memory_types=normalized_types,
                include_compressed=include_compressed,
            )
            filtered.sort(key=lambda chunk: chunk.relevance_score or 0.0, reverse=True)
            return filtered[:top_k]
        except MemoryLayerError as exc:
            if isinstance(exc, RetrievalError):
                raise
            raise RetrievalError(f"Failed to search memories: {exc}") from exc
        except Exception as exc:
            raise RetrievalError(f"Failed to search memories: {exc}") from exc

    def _filter_candidates(
        self,
        candidates: list[MemoryChunk],
        *,
        memory_types: tuple[MemoryType, ...] | None,
        include_compressed: bool,
    ) -> list[MemoryChunk]:
        """Apply retrieval candidate filters before ranking."""
        summarized_session_ids = {
            chunk.source_session_id
            for chunk in candidates
            if chunk.compression_source and chunk.source_session_id
        }

        filtered: list[MemoryChunk] = []
        for chunk in candidates:
            if chunk.importance < self._min_importance:
                continue
            if memory_types and chunk.memory_type not in memory_types:
                continue
            if chunk.compressed and chunk.session_id in summarized_session_ids:
                continue
            if chunk.compressed and not include_compressed:
                continue
            filtered.append(chunk)

        return filtered


__all__ = ["MemorySearcher"]
