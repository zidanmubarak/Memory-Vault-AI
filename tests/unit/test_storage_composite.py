from __future__ import annotations

from collections.abc import Sequence

import pytest

from memory_layer.models import MemoryChunk, MemoryType, PaginatedResult
from memory_layer.storage.base import (
    MemoryListQuery,
    MemorySearchQuery,
    MetadataStoreBackend,
    ProceduralMemoryRecord,
    SessionStatsRecord,
    VectorStoreBackend,
)
from memory_layer.storage.composite import CompositeStorage


class FakeVectorBackend(VectorStoreBackend):
    def __init__(self) -> None:
        self.chunks: dict[str, MemoryChunk] = {}

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def healthcheck(self) -> dict[str, str]:
        return {"status": "ok"}

    async def upsert_vectors(self, chunks: Sequence[MemoryChunk]) -> None:
        for chunk in chunks:
            self.chunks[chunk.id] = chunk

    async def query_vectors(self, query: MemorySearchQuery) -> list[MemoryChunk]:
        del query
        chunk = self.chunks.get("mem_a")
        if chunk is None:
            return []
        return [chunk.model_copy(update={"relevance_score": 0.91})]

    async def delete_vectors(self, memory_ids: Sequence[str], *, user_id: str) -> int:
        deleted = 0
        for memory_id in memory_ids:
            chunk = self.chunks.get(memory_id)
            if chunk is not None and chunk.user_id == user_id:
                self.chunks.pop(memory_id)
                deleted += 1
        return deleted

    async def delete_vectors_for_user(self, *, user_id: str) -> int:
        to_delete = [
            memory_id for memory_id, chunk in self.chunks.items() if chunk.user_id == user_id
        ]
        for memory_id in to_delete:
            self.chunks.pop(memory_id)
        return len(to_delete)


class FakeMetadataBackend(MetadataStoreBackend):
    def __init__(self) -> None:
        self.chunks: dict[str, MemoryChunk] = {}

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def healthcheck(self) -> dict[str, str]:
        return {"status": "ok"}

    async def upsert_memory_chunks(self, chunks: Sequence[MemoryChunk]) -> list[MemoryChunk]:
        for chunk in chunks:
            self.chunks[chunk.id] = chunk
        return list(chunks)

    async def get_memory_chunk(self, *, memory_id: str, user_id: str) -> MemoryChunk | None:
        chunk = self.chunks.get(memory_id)
        if chunk is None or chunk.user_id != user_id:
            return None
        return chunk

    async def list_memory_chunks(self, query: MemoryListQuery) -> PaginatedResult[MemoryChunk]:
        del query
        values = list(self.chunks.values())
        return PaginatedResult[MemoryChunk](items=values, total=len(values), page=1, page_size=20)

    async def delete_memory_chunk(self, *, memory_id: str, user_id: str) -> bool:
        chunk = self.chunks.get(memory_id)
        if chunk is None or chunk.user_id != user_id:
            return False
        self.chunks.pop(memory_id)
        return True

    async def delete_memory_chunks_for_user(self, *, user_id: str) -> int:
        to_delete = [
            memory_id for memory_id, chunk in self.chunks.items() if chunk.user_id == user_id
        ]
        for memory_id in to_delete:
            self.chunks.pop(memory_id)
        return len(to_delete)

    async def upsert_procedural_memory(
        self,
        record: ProceduralMemoryRecord,
    ) -> ProceduralMemoryRecord:
        return record

    async def list_procedural_memory(self, *, user_id: str) -> list[ProceduralMemoryRecord]:
        del user_id
        return []

    async def delete_procedural_memory(self, *, user_id: str, key: str) -> bool:
        del user_id, key
        return False

    async def upsert_session_stats(self, record: SessionStatsRecord) -> SessionStatsRecord:
        return record

    async def get_session_stats(self, *, session_id: str) -> SessionStatsRecord | None:
        del session_id
        return None


def _chunk(memory_id: str, user_id: str) -> MemoryChunk:
    return MemoryChunk(
        id=memory_id,
        user_id=user_id,
        session_id="sess_9a3b1c2d",
        memory_type=MemoryType.SEMANTIC,
        content="summary",
        importance=0.7,
        token_count=3,
        embedding=[0.1, 0.2],
    )


@pytest.mark.asyncio
async def test_composite_storage_upsert_and_query_are_coordinated() -> None:
    vector = FakeVectorBackend()
    metadata = FakeMetadataBackend()
    storage = CompositeStorage(vector_backend=vector, metadata_backend=metadata)

    await storage.initialize()
    await storage.upsert_memory_chunks([_chunk("mem_a", "user_a")])

    # SQLite-backed metadata does not persist embeddings, so query hydration must retain
    # candidate embeddings returned by the vector backend.
    metadata.chunks["mem_a"] = metadata.chunks["mem_a"].model_copy(update={"embedding": None})

    queried = await storage.query_vectors(
        MemorySearchQuery(user_id="user_a", query_embedding=[0.1, 0.2], top_k=5)
    )

    assert len(queried) == 1
    assert queried[0].id == "mem_a"
    assert queried[0].embedding == [0.1, 0.2]
    assert queried[0].relevance_score == 0.91


@pytest.mark.asyncio
async def test_composite_storage_delete_memory_chunk_removes_both_layers() -> None:
    vector = FakeVectorBackend()
    metadata = FakeMetadataBackend()
    storage = CompositeStorage(vector_backend=vector, metadata_backend=metadata)

    chunk = _chunk("mem_a", "user_a")
    await storage.upsert_memory_chunks([chunk])

    deleted = await storage.delete_memory_chunk(memory_id="mem_a", user_id="user_a")
    assert deleted is True

    queried = await storage.query_vectors(
        MemorySearchQuery(user_id="user_a", query_embedding=[0.1, 0.2], top_k=5)
    )
    assert queried == []

    health = await storage.healthcheck()
    assert health["status"] == "ok"
