from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, ClassVar, cast

import pytest

from memory_layer.compression import CompressionResult
from memory_layer.exceptions import ConfigurationError
from memory_layer.models import MemoryChunk, MemoryConfig, MemoryType, PaginatedResult, RecallResult
from memory_layer.sdk import MemoryLayer
from memory_layer.storage.base import (
    MemoryListQuery,
    MemorySearchQuery,
    ProceduralMemoryRecord,
    SessionStatsRecord,
    StorageBackend,
)


class StubIngestionEngine:
    def __init__(self, result: list[MemoryChunk]) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    async def ingest(
        self,
        text: str,
        user_id: str,
        session_id: str,
        memory_type_hint: MemoryType | None = None,
    ) -> list[MemoryChunk]:
        self.calls.append(
            {
                "text": text,
                "user_id": user_id,
                "session_id": session_id,
                "memory_type_hint": memory_type_hint,
            }
        )
        return [chunk.model_copy() for chunk in self.result]


class StubRetrievalEngine:
    def __init__(self, result: RecallResult) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    async def recall(
        self,
        query: str,
        user_id: str,
        *,
        top_k: int = 5,
        token_budget: int = 2000,
        memory_types: Sequence[MemoryType] | None = None,
        include_compressed: bool = False,
        reranker_enabled: bool | None = None,
    ) -> RecallResult:
        self.calls.append(
            {
                "query": query,
                "user_id": user_id,
                "top_k": top_k,
                "token_budget": token_budget,
                "memory_types": tuple(memory_types) if memory_types else None,
                "include_compressed": include_compressed,
                "reranker_enabled": reranker_enabled,
            }
        )
        return self.result.model_copy()


class FakeStorage(StorageBackend):
    def __init__(self) -> None:
        self.initialized = False
        self.closed = False
        self.list_result = PaginatedResult[MemoryChunk](items=[], total=0, page=1, page_size=20)
        self.list_query: MemoryListQuery | None = None
        self.deleted_chunk_ids: list[str] = []
        self.deleted_user_calls = 0
        self.procedural_records: list[ProceduralMemoryRecord] = []
        self.deleted_procedural_keys: list[str] = []
        self.session_stats: dict[str, SessionStatsRecord] = {}

    async def initialize(self) -> None:
        self.initialized = True

    async def close(self) -> None:
        self.closed = True

    async def healthcheck(self) -> dict[str, str]:
        return {"status": "ok"}

    async def upsert_vectors(self, chunks: Sequence[MemoryChunk]) -> None:
        del chunks

    async def query_vectors(self, query: MemorySearchQuery) -> list[MemoryChunk]:
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
        self.list_query = query
        return self.list_result

    async def delete_memory_chunk(self, *, memory_id: str, user_id: str) -> bool:
        del user_id
        self.deleted_chunk_ids.append(memory_id)
        return True

    async def delete_memory_chunks_for_user(self, *, user_id: str) -> int:
        del user_id
        self.deleted_user_calls += 1
        return 3

    async def upsert_procedural_memory(
        self,
        record: ProceduralMemoryRecord,
    ) -> ProceduralMemoryRecord:
        self.procedural_records.append(record)
        return record

    async def list_procedural_memory(self, *, user_id: str) -> list[ProceduralMemoryRecord]:
        return [record for record in self.procedural_records if record.user_id == user_id]

    async def delete_procedural_memory(self, *, user_id: str, key: str) -> bool:
        del user_id
        self.deleted_procedural_keys.append(key)
        return True

    async def upsert_session_stats(self, record: SessionStatsRecord) -> SessionStatsRecord:
        self.session_stats[record.session_id] = record
        return record

    async def get_session_stats(self, *, session_id: str) -> SessionStatsRecord | None:
        return self.session_stats.get(session_id)


def _chunk(memory_id: str) -> MemoryChunk:
    now = datetime(2026, 4, 19, 10, 0, tzinfo=UTC)
    return MemoryChunk(
        id=memory_id,
        user_id="user_a",
        session_id="sess_a",
        memory_type=MemoryType.SEMANTIC,
        content="content",
        importance=0.8,
        token_count=1,
        embedding=[0.1, 0.2],
        created_at=now,
        updated_at=now,
    )


def test_memory_layer_requires_user_id() -> None:
    with pytest.raises(ValueError):
        MemoryLayer(user_id="")


def test_memory_layer_supports_qdrant_backend() -> None:
    sdk = MemoryLayer(
        user_id="user_a",
        config=MemoryConfig(
            storage_backend="qdrant",
            qdrant_url="http://localhost:6333",
            metadata_backend="sqlite",
        ),
    )
    vector_backend = cast(Any, sdk._storage)._vector_backend
    assert vector_backend.__class__.__name__ == "QdrantAdapter"


def test_memory_layer_rejects_unsupported_metadata_backend() -> None:
    with pytest.raises(ConfigurationError):
        MemoryLayer(
            user_id="user_a",
            config=MemoryConfig(
                storage_backend="chroma",
                metadata_backend="postgres",
                postgres_url="postgresql://localhost/test",
            ),
        )


@pytest.mark.asyncio
async def test_save_delegates_to_ingestion_engine() -> None:
    storage = FakeStorage()
    ingestion = StubIngestionEngine([_chunk("mem_a")])
    retrieval = StubRetrievalEngine(RecallResult())
    sdk = MemoryLayer(
        user_id="user_a",
        session_id="sess_1234",
        storage=storage,
        ingestion_engine=ingestion,
        retrieval_engine=retrieval,
        config=MemoryConfig(),
    )

    saved = await sdk.save("hello world", memory_type_hint=MemoryType.SEMANTIC)

    assert [chunk.id for chunk in saved] == ["mem_a"]
    assert ingestion.calls[0]["session_id"] == "sess_1234"
    assert storage.initialized is True


@pytest.mark.asyncio
async def test_recall_uses_config_defaults() -> None:
    storage = FakeStorage()
    ingestion = StubIngestionEngine([])
    retrieval = StubRetrievalEngine(
        RecallResult(memories=[_chunk("mem_a")], total_tokens=3, budget_used=0.3)
    )
    sdk = MemoryLayer(
        user_id="user_a",
        storage=storage,
        ingestion_engine=ingestion,
        retrieval_engine=retrieval,
        config=MemoryConfig(top_k=7, token_budget=1234),
    )

    result = await sdk.recall("query")

    assert [chunk.id for chunk in result.memories] == ["mem_a"]
    assert retrieval.calls[0]["top_k"] == 7
    assert retrieval.calls[0]["token_budget"] == 1234


@pytest.mark.asyncio
async def test_list_builds_query_with_user_scope() -> None:
    storage = FakeStorage()
    ingestion = StubIngestionEngine([])
    retrieval = StubRetrievalEngine(RecallResult())
    storage.list_result = PaginatedResult[MemoryChunk](
        items=[_chunk("mem_a")],
        total=1,
        page=2,
        page_size=10,
    )
    sdk = MemoryLayer(
        user_id="user_a",
        storage=storage,
        ingestion_engine=ingestion,
        retrieval_engine=retrieval,
        config=MemoryConfig(),
    )

    result = await sdk.list(memory_type=MemoryType.SEMANTIC, page=2, page_size=10)

    assert result.total == 1
    assert storage.list_query is not None
    assert storage.list_query.user_id == "user_a"
    assert storage.list_query.memory_type is MemoryType.SEMANTIC


@pytest.mark.asyncio
async def test_procedural_memory_read_write_behaviors() -> None:
    storage = FakeStorage()
    sdk = MemoryLayer(
        user_id="user_a",
        storage=storage,
        ingestion_engine=StubIngestionEngine([]),
        retrieval_engine=StubRetrievalEngine(RecallResult()),
        config=MemoryConfig(),
    )

    upserted = await sdk.upsert_procedural_memory(
        key="tone",
        value="Use concise technical responses.",
        confidence=0.85,
        source_chunk_id="mem_source_1",
    )
    assert upserted.user_id == "user_a"
    assert upserted.key == "tone"
    assert upserted.value == "Use concise technical responses."
    assert upserted.confidence == pytest.approx(0.85)
    assert upserted.source_chunk_id == "mem_source_1"

    listed = await sdk.list_procedural_memory()
    assert [record.key for record in listed] == ["tone"]

    deleted = await sdk.delete_procedural_memory(key="tone")
    assert deleted is True
    assert storage.deleted_procedural_keys == ["tone"]


@pytest.mark.asyncio
async def test_compress_delegates_to_compressor(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class FakeCompressor:
        init_calls: ClassVar[list[dict[str, object]]] = []
        compress_calls: ClassVar[list[dict[str, object]]] = []

        def __init__(self, *, storage: StorageBackend, compression_threshold: int) -> None:
            FakeCompressor.init_calls.append(
                {
                    "storage": storage,
                    "compression_threshold": compression_threshold,
                }
            )

        async def compress_user(
            self,
            user_id: str,
            *,
            force: bool = False,
            sessions_to_compress: int | None = None,
        ) -> CompressionResult:
            FakeCompressor.compress_calls.append(
                {
                    "user_id": user_id,
                    "force": force,
                    "sessions_to_compress": sessions_to_compress,
                }
            )
            return CompressionResult(
                user_id=user_id,
                total_uncompressed_sessions=4,
                sessions_compressed=2,
                summaries_created=2,
                memories_marked_compressed=6,
                compressed_session_ids=("sess_1", "sess_2"),
            )

    monkeypatch.setattr("memory_layer.sdk.MemoryCompressor", FakeCompressor)

    storage = FakeStorage()
    sdk = MemoryLayer(
        user_id="user_a",
        storage=storage,
        ingestion_engine=StubIngestionEngine([]),
        retrieval_engine=StubRetrievalEngine(RecallResult()),
        config=MemoryConfig(compression_threshold=7),
    )

    result = await sdk.compress(force=True, sessions_to_compress=2)
    assert result.sessions_compressed == 2
    assert result.compressed_session_ids == ("sess_1", "sess_2")
    assert FakeCompressor.init_calls == [
        {
            "storage": storage,
            "compression_threshold": 7,
        }
    ]
    assert FakeCompressor.compress_calls == [
        {
            "user_id": "user_a",
            "force": True,
            "sessions_to_compress": 2,
        }
    ]


@pytest.mark.asyncio
async def test_forget_and_forget_all_behaviors() -> None:
    storage = FakeStorage()
    storage.procedural_records = [
        ProceduralMemoryRecord(user_id="user_a", key="k1", value="v1"),
        ProceduralMemoryRecord(user_id="user_a", key="k2", value="v2"),
    ]
    sdk = MemoryLayer(
        user_id="user_a",
        storage=storage,
        ingestion_engine=StubIngestionEngine([]),
        retrieval_engine=StubRetrievalEngine(RecallResult()),
        config=MemoryConfig(),
    )

    deleted = await sdk.forget(memory_id="mem_x")
    assert deleted is True
    assert storage.deleted_chunk_ids == ["mem_x"]

    with pytest.raises(ValueError):
        await sdk.forget_all(confirm=False)

    total_deleted = await sdk.forget_all(confirm=True)
    assert total_deleted == 5
    assert sorted(storage.deleted_procedural_keys) == ["k1", "k2"]


@pytest.mark.asyncio
async def test_end_session_and_context_manager() -> None:
    storage = FakeStorage()
    session_id = "sess_abcd1234"
    storage.session_stats[session_id] = SessionStatsRecord(
        session_id=session_id,
        user_id="user_a",
        memory_count=2,
        total_tokens_stored=12,
        started_at=datetime(2026, 4, 19, 9, 0, tzinfo=UTC),
        last_activity=datetime(2026, 4, 19, 9, 30, tzinfo=UTC),
    )
    sdk = MemoryLayer(
        user_id="user_a",
        session_id=session_id,
        storage=storage,
        ingestion_engine=StubIngestionEngine([]),
        retrieval_engine=StubRetrievalEngine(RecallResult()),
        config=MemoryConfig(),
    )

    ended = await sdk.end_session()
    assert ended is not None
    assert ended.ended_at is not None

    async with sdk:
        pass

    assert storage.closed is True


@pytest.mark.asyncio
async def test_end_session_returns_none_for_foreign_session_owner() -> None:
    storage = FakeStorage()
    session_id = "sess_abcd1234"
    storage.session_stats[session_id] = SessionStatsRecord(
        session_id=session_id,
        user_id="user_other",
        memory_count=2,
        total_tokens_stored=12,
        started_at=datetime(2026, 4, 19, 9, 0, tzinfo=UTC),
        last_activity=datetime(2026, 4, 19, 9, 30, tzinfo=UTC),
    )
    sdk = MemoryLayer(
        user_id="user_a",
        session_id=session_id,
        storage=storage,
        ingestion_engine=StubIngestionEngine([]),
        retrieval_engine=StubRetrievalEngine(RecallResult()),
        config=MemoryConfig(),
    )

    ended = await sdk.end_session()

    assert ended is None
    assert storage.session_stats[session_id].user_id == "user_other"
