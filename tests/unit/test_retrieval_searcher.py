from __future__ import annotations

from collections.abc import Sequence

import pytest

from memory_layer.exceptions import RetrievalError, StorageError
from memory_layer.models import MemoryChunk, MemoryType, PaginatedResult
from memory_layer.retrieval.searcher import MemorySearcher
from memory_layer.storage.base import (
    MemoryListQuery,
    MemorySearchQuery,
    ProceduralMemoryRecord,
    SessionStatsRecord,
    StorageBackend,
)


class FakeStorage(StorageBackend):
    def __init__(self) -> None:
        self.query_results: list[MemoryChunk] = []
        self.last_query: MemorySearchQuery | None = None
        self.raise_on_query = False

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def healthcheck(self) -> dict[str, str]:
        return {"status": "ok"}

    async def upsert_vectors(self, chunks: Sequence[MemoryChunk]) -> None:
        del chunks
        return None

    async def query_vectors(self, query: MemorySearchQuery) -> list[MemoryChunk]:
        self.last_query = query
        if self.raise_on_query:
            raise StorageError("vector backend failed")
        return [chunk.model_copy() for chunk in self.query_results]

    async def delete_vectors(self, memory_ids: Sequence[str], *, user_id: str) -> int:
        del memory_ids, user_id
        return 0

    async def delete_vectors_for_user(self, *, user_id: str) -> int:
        del user_id
        return 0

    async def upsert_memory_chunks(self, chunks: Sequence[MemoryChunk]) -> list[MemoryChunk]:
        return list(chunks)

    async def get_memory_chunk(self, *, memory_id: str, user_id: str) -> MemoryChunk | None:
        del memory_id, user_id
        return None

    async def list_memory_chunks(self, query: MemoryListQuery) -> PaginatedResult[MemoryChunk]:
        del query
        return PaginatedResult[MemoryChunk](items=[], total=0, page=1, page_size=20)

    async def delete_memory_chunk(self, *, memory_id: str, user_id: str) -> bool:
        del memory_id, user_id
        return False

    async def delete_memory_chunks_for_user(self, *, user_id: str) -> int:
        del user_id
        return 0

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


def _chunk(
    *,
    memory_id: str,
    memory_type: MemoryType,
    importance: float,
    relevance_score: float,
    compressed: bool = False,
    compression_source: bool = False,
    session_id: str = "sess_1",
    source_session_id: str | None = None,
) -> MemoryChunk:
    return MemoryChunk(
        id=memory_id,
        user_id="user_a",
        session_id=session_id,
        memory_type=memory_type,
        content=f"content-{memory_id}",
        importance=importance,
        token_count=4,
        embedding=[0.1, 0.2],
        compressed=compressed,
        compression_source=compression_source,
        source_session_id=source_session_id,
        relevance_score=relevance_score,
    )


def test_searcher_validates_constructor_inputs() -> None:
    with pytest.raises(ValueError):
        MemorySearcher(storage=FakeStorage(), candidate_multiplier=0)
    with pytest.raises(ValueError):
        MemorySearcher(storage=FakeStorage(), min_importance=-0.1)


@pytest.mark.asyncio
async def test_search_validates_request_inputs() -> None:
    searcher = MemorySearcher(storage=FakeStorage())

    with pytest.raises(ValueError):
        await searcher.search(user_id="", query_embedding=[0.1], top_k=5)
    with pytest.raises(ValueError):
        await searcher.search(user_id="user_a", query_embedding=[], top_k=5)
    with pytest.raises(ValueError):
        await searcher.search(user_id="user_a", query_embedding=[0.1], top_k=0)


@pytest.mark.asyncio
async def test_search_expands_candidate_pool_and_returns_top_k_ordered() -> None:
    storage = FakeStorage()
    storage.query_results = [
        _chunk(
            memory_id="mem_low",
            memory_type=MemoryType.SEMANTIC,
            importance=0.8,
            relevance_score=0.2,
        ),
        _chunk(
            memory_id="mem_high",
            memory_type=MemoryType.SEMANTIC,
            importance=0.8,
            relevance_score=0.95,
        ),
        _chunk(
            memory_id="mem_mid",
            memory_type=MemoryType.SEMANTIC,
            importance=0.8,
            relevance_score=0.7,
        ),
    ]
    searcher = MemorySearcher(storage=storage, candidate_multiplier=4)

    results = await searcher.search(
        user_id="user_a",
        query_embedding=[0.2, 0.8],
        top_k=2,
    )

    assert [chunk.id for chunk in results] == ["mem_high", "mem_mid"]
    assert storage.last_query is not None
    assert storage.last_query.top_k == 8
    assert storage.last_query.include_compressed is True
    assert storage.last_query.min_importance == 0.0


@pytest.mark.asyncio
async def test_search_filters_min_importance_and_memory_type() -> None:
    storage = FakeStorage()
    storage.query_results = [
        _chunk(
            memory_id="mem_semantic",
            memory_type=MemoryType.SEMANTIC,
            importance=0.9,
            relevance_score=0.8,
        ),
        _chunk(
            memory_id="mem_low_importance",
            memory_type=MemoryType.SEMANTIC,
            importance=0.1,
            relevance_score=0.99,
        ),
        _chunk(
            memory_id="mem_episodic",
            memory_type=MemoryType.EPISODIC,
            importance=0.9,
            relevance_score=0.85,
        ),
    ]
    searcher = MemorySearcher(storage=storage)

    results = await searcher.search(
        user_id="user_a",
        query_embedding=[0.2, 0.8],
        top_k=5,
        memory_types=[MemoryType.SEMANTIC],
    )

    assert [chunk.id for chunk in results] == ["mem_semantic"]
    assert storage.last_query is not None
    assert storage.last_query.memory_types == (MemoryType.SEMANTIC,)


@pytest.mark.asyncio
async def test_search_excludes_compressed_chunk_when_summary_exists() -> None:
    storage = FakeStorage()
    storage.query_results = [
        _chunk(
            memory_id="mem_summary",
            memory_type=MemoryType.SEMANTIC,
            importance=0.9,
            relevance_score=0.6,
            compression_source=True,
            source_session_id="sess_old",
            session_id="sess_new",
        ),
        _chunk(
            memory_id="mem_compressed_original",
            memory_type=MemoryType.EPISODIC,
            importance=0.9,
            relevance_score=0.95,
            compressed=True,
            session_id="sess_old",
        ),
    ]
    searcher = MemorySearcher(storage=storage)

    results = await searcher.search(
        user_id="user_a",
        query_embedding=[0.1],
        top_k=5,
        include_compressed=True,
    )

    assert [chunk.id for chunk in results] == ["mem_summary"]


@pytest.mark.asyncio
async def test_search_include_compressed_flag_controls_unsummarized_chunks() -> None:
    storage = FakeStorage()
    storage.query_results = [
        _chunk(
            memory_id="mem_compressed",
            memory_type=MemoryType.EPISODIC,
            importance=0.9,
            relevance_score=0.8,
            compressed=True,
            session_id="sess_x",
        )
    ]
    searcher = MemorySearcher(storage=storage)

    excluded = await searcher.search(
        user_id="user_a",
        query_embedding=[0.1],
        top_k=5,
        include_compressed=False,
    )
    included = await searcher.search(
        user_id="user_a",
        query_embedding=[0.1],
        top_k=5,
        include_compressed=True,
    )

    assert excluded == []
    assert [chunk.id for chunk in included] == ["mem_compressed"]


@pytest.mark.asyncio
async def test_search_wraps_storage_errors() -> None:
    storage = FakeStorage()
    storage.raise_on_query = True
    searcher = MemorySearcher(storage=storage)

    with pytest.raises(RetrievalError):
        await searcher.search(
            user_id="user_a",
            query_embedding=[0.1, 0.2],
            top_k=5,
        )
