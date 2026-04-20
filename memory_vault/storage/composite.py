from __future__ import annotations

from collections.abc import Sequence

from memory_vault.exceptions import StorageError
from memory_vault.models import MemoryChunk, PaginatedResult
from memory_vault.storage.base import (
    MemoryListQuery,
    MemorySearchQuery,
    MetadataStoreBackend,
    ProceduralMemoryRecord,
    SessionStatsRecord,
    StorageBackend,
    VectorStoreBackend,
)


class CompositeStorage(StorageBackend):
    """Coordinates vector and metadata stores behind one storage interface."""

    def __init__(
        self,
        *,
        vector_backend: VectorStoreBackend,
        metadata_backend: MetadataStoreBackend,
    ) -> None:
        self._vector_backend = vector_backend
        self._metadata_backend = metadata_backend

    async def initialize(self) -> None:
        """Initialize both underlying storage backends."""
        await self._metadata_backend.initialize()
        await self._vector_backend.initialize()

    async def close(self) -> None:
        """Close both backends and preserve the first raised error."""
        first_error: BaseException | None = None

        try:
            await self._vector_backend.close()
        except Exception as exc:  # pragma: no cover - defensive error wrapping
            first_error = exc

        try:
            await self._metadata_backend.close()
        except Exception as exc:  # pragma: no cover - defensive error wrapping
            if first_error is None:
                first_error = exc

        if first_error is not None:
            raise StorageError(f"Failed to close composite storage: {first_error}") from first_error

    async def healthcheck(self) -> dict[str, str]:
        """Return combined health signals for vector and metadata backends."""
        metadata = await self._metadata_backend.healthcheck()
        vector = await self._vector_backend.healthcheck()

        metadata_status = metadata.get("status", "unknown")
        vector_status = vector.get("status", "unknown")
        status = "ok" if metadata_status == "ok" and vector_status == "ok" else "degraded"

        return {
            "status": status,
            "metadata": metadata_status,
            "vector": vector_status,
        }

    async def upsert_vectors(self, chunks: Sequence[MemoryChunk]) -> None:
        """Delegate vector upserts directly to vector backend."""
        await self._vector_backend.upsert_vectors(chunks)

    async def query_vectors(self, query: MemorySearchQuery) -> list[MemoryChunk]:
        """Query vectors then hydrate canonical chunk records from metadata backend."""
        candidates = await self._vector_backend.query_vectors(query)
        hydrated: list[MemoryChunk] = []

        for candidate in candidates:
            stored = await self._metadata_backend.get_memory_chunk(
                memory_id=candidate.id,
                user_id=query.user_id,
            )
            if stored is None:
                continue
            hydrated.append(
                stored.model_copy(
                    update={
                        "embedding": candidate.embedding,
                        "relevance_score": candidate.relevance_score,
                    }
                )
            )

        return hydrated

    async def delete_vectors(self, memory_ids: Sequence[str], *, user_id: str) -> int:
        """Delegate vector deletes directly to vector backend."""
        return await self._vector_backend.delete_vectors(memory_ids, user_id=user_id)

    async def delete_vectors_for_user(self, *, user_id: str) -> int:
        """Delegate vector deletes for one user directly to vector backend."""
        return await self._vector_backend.delete_vectors_for_user(user_id=user_id)

    async def upsert_memory_chunks(self, chunks: Sequence[MemoryChunk]) -> list[MemoryChunk]:
        """Persist metadata and corresponding vector records."""
        persisted = await self._metadata_backend.upsert_memory_chunks(chunks)
        try:
            await self._vector_backend.upsert_vectors(persisted)
        except Exception as exc:
            raise StorageError(
                f"Failed to synchronize vector upsert after metadata write: {exc}"
            ) from exc
        return persisted

    async def get_memory_chunk(self, *, memory_id: str, user_id: str) -> MemoryChunk | None:
        """Delegate metadata lookups directly to metadata backend."""
        return await self._metadata_backend.get_memory_chunk(memory_id=memory_id, user_id=user_id)

    async def list_memory_chunks(self, query: MemoryListQuery) -> PaginatedResult[MemoryChunk]:
        """Delegate memory list requests directly to metadata backend."""
        return await self._metadata_backend.list_memory_chunks(query)

    async def delete_memory_chunk(self, *, memory_id: str, user_id: str) -> bool:
        """Delete one memory chunk from vector and metadata stores."""
        existing = await self._metadata_backend.get_memory_chunk(
            memory_id=memory_id,
            user_id=user_id,
        )
        if existing is None:
            return False

        await self._vector_backend.delete_vectors([memory_id], user_id=user_id)
        return await self._metadata_backend.delete_memory_chunk(
            memory_id=memory_id,
            user_id=user_id,
        )

    async def delete_memory_chunks_for_user(self, *, user_id: str) -> int:
        """Delete all user data from both vector and metadata stores."""
        await self._vector_backend.delete_vectors_for_user(user_id=user_id)
        return await self._metadata_backend.delete_memory_chunks_for_user(user_id=user_id)

    async def upsert_procedural_memory(
        self,
        record: ProceduralMemoryRecord,
    ) -> ProceduralMemoryRecord:
        """Delegate procedural memory upsert to metadata backend."""
        return await self._metadata_backend.upsert_procedural_memory(record)

    async def list_procedural_memory(self, *, user_id: str) -> list[ProceduralMemoryRecord]:
        """Delegate procedural memory listing to metadata backend."""
        return await self._metadata_backend.list_procedural_memory(user_id=user_id)

    async def delete_procedural_memory(self, *, user_id: str, key: str) -> bool:
        """Delegate procedural memory delete to metadata backend."""
        return await self._metadata_backend.delete_procedural_memory(user_id=user_id, key=key)

    async def upsert_session_stats(self, record: SessionStatsRecord) -> SessionStatsRecord:
        """Delegate session stats upsert to metadata backend."""
        return await self._metadata_backend.upsert_session_stats(record)

    async def get_session_stats(self, *, session_id: str) -> SessionStatsRecord | None:
        """Delegate session stats lookup to metadata backend."""
        return await self._metadata_backend.get_session_stats(session_id=session_id)


__all__ = ["CompositeStorage"]
