from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from memory_layer.exceptions import BudgetExceededError, RetrievalError
from memory_layer.models import MemoryChunk, MemoryType, PaginatedResult
from memory_layer.retrieval.engine import RetrievalEngine
from memory_layer.storage.base import (
    MemoryListQuery,
    ProceduralMemoryRecord,
    SessionStatsRecord,
    StorageBackend,
)


class StubEmbedder:
    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = vectors
        self.calls: list[list[str]] = []

    async def encode_batch(self, chunks: list[str]) -> list[list[float]]:
        self.calls.append(list(chunks))
        return [vector[:] for vector in self._vectors]


class StubSearcher:
    def __init__(self, results: list[MemoryChunk]) -> None:
        self._results = results
        self.calls: list[dict[str, object]] = []
        self.raise_error = False

    async def search(
        self,
        *,
        user_id: str,
        query_embedding: Sequence[float],
        top_k: int = 5,
        memory_types: Sequence[MemoryType] | None = None,
        include_compressed: bool = False,
    ) -> list[MemoryChunk]:
        if self.raise_error:
            raise RuntimeError("search failed")

        self.calls.append(
            {
                "user_id": user_id,
                "query_embedding": list(query_embedding),
                "top_k": top_k,
                "memory_types": tuple(memory_types) if memory_types else None,
                "include_compressed": include_compressed,
            }
        )
        return [chunk.model_copy() for chunk in self._results]


class StubReranker:
    def __init__(self, results: list[MemoryChunk] | None = None) -> None:
        self._results = results
        self.calls: list[dict[str, object]] = []

    def rerank(
        self,
        candidates: Sequence[MemoryChunk],
        *,
        top_k: int | None = None,
        query_text: str | None = None,
    ) -> list[MemoryChunk]:
        self.calls.append(
            {
                "candidate_ids": [chunk.id for chunk in candidates],
                "top_k": top_k,
                "query_text": query_text,
            }
        )
        if self._results is None:
            return list(candidates)
        return [chunk.model_copy() for chunk in self._results]


class FakeStorage(StorageBackend):
    def __init__(self) -> None:
        self.procedural_records: list[ProceduralMemoryRecord] = []

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def healthcheck(self) -> dict[str, str]:
        return {"status": "ok"}

    async def upsert_vectors(self, chunks: Sequence[MemoryChunk]) -> None:
        del chunks
        return None

    async def query_vectors(self, query):  # type: ignore[no-untyped-def]
        del query
        return []

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
        self.procedural_records.append(record)
        return record

    async def list_procedural_memory(self, *, user_id: str) -> list[ProceduralMemoryRecord]:
        return [record for record in self.procedural_records if record.user_id == user_id]

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
    content: str,
    importance: float,
    relevance_score: float,
) -> MemoryChunk:
    now = datetime(2026, 4, 19, 10, 0, tzinfo=UTC)
    return MemoryChunk(
        id=memory_id,
        user_id="user_a",
        session_id="sess_a",
        memory_type=memory_type,
        content=content,
        importance=importance,
        token_count=1,
        embedding=[0.1, 0.2],
        relevance_score=relevance_score,
        created_at=now,
        updated_at=now,
    )


def _procedural_record(*, key: str, value: str) -> ProceduralMemoryRecord:
    return ProceduralMemoryRecord(
        user_id="user_a",
        key=key,
        value=value,
        confidence=0.9,
        updated_at=datetime(2026, 4, 19, 10, 30, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_recall_validates_required_inputs() -> None:
    engine = RetrievalEngine(
        storage=FakeStorage(),
        embedder=StubEmbedder([[0.1, 0.2]]),
        searcher=StubSearcher([]),
    )

    with pytest.raises(ValueError):
        await engine.recall("", "user_a")
    with pytest.raises(ValueError):
        await engine.recall("query", "")
    with pytest.raises(ValueError):
        await engine.recall("query", "user_a", top_k=0)
    with pytest.raises(ValueError):
        await engine.recall("query", "user_a", token_budget=0)


@pytest.mark.asyncio
async def test_recall_uses_embedder_searcher_and_reranker() -> None:
    storage = FakeStorage()
    embedder = StubEmbedder([[0.7, 0.3]])
    searcher = StubSearcher(
        [
            _chunk(
                memory_id="mem_a",
                memory_type=MemoryType.SEMANTIC,
                content="semantic content",
                importance=0.8,
                relevance_score=0.6,
            )
        ]
    )
    reranker = StubReranker(
        [
            _chunk(
                memory_id="mem_reranked",
                memory_type=MemoryType.SEMANTIC,
                content="reranked content",
                importance=0.9,
                relevance_score=0.95,
            )
        ]
    )
    engine = RetrievalEngine(
        storage=storage,
        embedder=embedder,
        searcher=searcher,
        reranker=reranker,
        reranker_enabled=True,
        candidate_pool_multiplier=4,
        token_counter=lambda text: len(text.split()),
    )

    result = await engine.recall(
        "Find important memory",
        "user_a",
        top_k=2,
        token_budget=20,
    )

    assert embedder.calls == [["Find important memory"]]
    assert len(searcher.calls) == 1
    assert searcher.calls[0]["top_k"] == 8
    assert searcher.calls[0]["query_embedding"] == [0.7, 0.3]
    assert len(reranker.calls) == 1
    assert reranker.calls[0]["query_text"] == "Find important memory"
    assert [chunk.id for chunk in result.memories] == ["mem_reranked"]
    assert result.prompt_block == "<memory>\n[SEMANTIC] reranked content\n</memory>"


@pytest.mark.asyncio
async def test_recall_includes_procedural_memories_first() -> None:
    storage = FakeStorage()
    storage.procedural_records.append(
        _procedural_record(key="communication_style", value="Be concise and technical")
    )
    engine = RetrievalEngine(
        storage=storage,
        embedder=StubEmbedder([[0.2, 0.8]]),
        searcher=StubSearcher(
            [
                _chunk(
                    memory_id="mem_semantic",
                    memory_type=MemoryType.SEMANTIC,
                    content="Semantic memory content",
                    importance=0.8,
                    relevance_score=0.7,
                )
            ]
        ),
        reranker_enabled=False,
        token_counter=lambda text: len(text.split()),
    )

    result = await engine.recall("query", "user_a", top_k=2, token_budget=20)

    assert len(result.memories) == 2
    assert result.memories[0].memory_type is MemoryType.PROCEDURAL
    assert result.memories[1].memory_type is MemoryType.SEMANTIC


@pytest.mark.asyncio
async def test_recall_respects_procedural_only_filter() -> None:
    storage = FakeStorage()
    storage.procedural_records.append(
        _procedural_record(key="style", value="Answer with examples")
    )
    searcher = StubSearcher([])
    engine = RetrievalEngine(
        storage=storage,
        embedder=StubEmbedder([[0.5, 0.5]]),
        searcher=searcher,
        token_counter=lambda text: len(text.split()),
    )

    result = await engine.recall(
        "query",
        "user_a",
        top_k=3,
        token_budget=20,
        memory_types=[MemoryType.PROCEDURAL],
    )

    assert [chunk.memory_type for chunk in result.memories] == [MemoryType.PROCEDURAL]
    assert searcher.calls == []


@pytest.mark.asyncio
async def test_recall_enforces_token_budget_and_top_k() -> None:
    storage = FakeStorage()
    searcher = StubSearcher(
        [
            _chunk(
                memory_id="mem_short",
                memory_type=MemoryType.SEMANTIC,
                content="short content",
                importance=0.7,
                relevance_score=0.9,
            ),
            _chunk(
                memory_id="mem_long",
                memory_type=MemoryType.SEMANTIC,
                content="this chunk has too many tokens for tiny budget",
                importance=0.8,
                relevance_score=0.8,
            ),
        ]
    )
    engine = RetrievalEngine(
        storage=storage,
        embedder=StubEmbedder([[0.1, 0.9]]),
        searcher=searcher,
        token_counter=lambda text: len(text.split()),
    )

    result = await engine.recall("query", "user_a", top_k=1, token_budget=2)

    assert [chunk.id for chunk in result.memories] == ["mem_short"]
    assert result.total_tokens == 2
    assert result.budget_used == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_recall_raises_budget_exceeded_when_nothing_fits() -> None:
    storage = FakeStorage()
    searcher = StubSearcher(
        [
            _chunk(
                memory_id="mem_long",
                memory_type=MemoryType.SEMANTIC,
                content="this candidate cannot fit the budget",
                importance=0.8,
                relevance_score=0.9,
            )
        ]
    )
    engine = RetrievalEngine(
        storage=storage,
        embedder=StubEmbedder([[0.5, 0.5]]),
        searcher=searcher,
        token_counter=lambda text: len(text.split()),
    )

    with pytest.raises(BudgetExceededError):
        await engine.recall("query", "user_a", top_k=1, token_budget=2)


@pytest.mark.asyncio
async def test_recall_wraps_unexpected_errors() -> None:
    searcher = StubSearcher([])
    searcher.raise_error = True
    engine = RetrievalEngine(
        storage=FakeStorage(),
        embedder=StubEmbedder([[0.1, 0.2]]),
        searcher=searcher,
    )

    with pytest.raises(RetrievalError):
        await engine.recall("query", "user_a")
