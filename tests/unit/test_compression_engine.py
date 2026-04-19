from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from memory_layer.compression.engine import HeuristicSessionSummarizer, MemoryCompressor
from memory_layer.exceptions import CompressionError
from memory_layer.models import MemoryChunk, MemoryType, PaginatedResult
from memory_layer.storage.base import (
    MemoryListQuery,
    MemorySearchQuery,
    ProceduralMemoryRecord,
    SessionStatsRecord,
    StorageBackend,
)


class FakeStorage(StorageBackend):
    def __init__(self) -> None:
        self.chunks: dict[str, MemoryChunk] = {}
        self.procedural: list[ProceduralMemoryRecord] = []
        self.sessions: dict[str, SessionStatsRecord] = {}
        self.raise_on_list = False

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

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
        for chunk in chunks:
            self.chunks[chunk.id] = chunk
        return list(chunks)

    async def get_memory_chunk(self, *, memory_id: str, user_id: str) -> MemoryChunk | None:
        chunk = self.chunks.get(memory_id)
        if chunk is None or chunk.user_id != user_id:
            return None
        return chunk

    async def list_memory_chunks(self, query: MemoryListQuery) -> PaginatedResult[MemoryChunk]:
        if self.raise_on_list:
            raise RuntimeError("boom")

        items = [
            chunk
            for chunk in self.chunks.values()
            if chunk.user_id == query.user_id
            and (query.memory_type is None or chunk.memory_type is query.memory_type)
            and (query.include_compressed or not chunk.compressed)
        ]
        items.sort(key=lambda chunk: chunk.created_at, reverse=True)
        total = len(items)
        start = (query.page - 1) * query.page_size
        end = start + query.page_size
        paged = items[start:end]
        return PaginatedResult[MemoryChunk](
            items=paged,
            total=total,
            page=query.page,
            page_size=query.page_size,
        )

    async def delete_memory_chunk(self, *, memory_id: str, user_id: str) -> bool:
        chunk = self.chunks.get(memory_id)
        if chunk is None or chunk.user_id != user_id:
            return False
        self.chunks.pop(memory_id)
        return True

    async def delete_memory_chunks_for_user(self, *, user_id: str) -> int:
        to_delete = [
            memory_id
            for memory_id, chunk in self.chunks.items()
            if chunk.user_id == user_id
        ]
        for memory_id in to_delete:
            self.chunks.pop(memory_id)
        return len(to_delete)

    async def upsert_procedural_memory(
        self,
        record: ProceduralMemoryRecord,
    ) -> ProceduralMemoryRecord:
        self.procedural.append(record)
        return record

    async def list_procedural_memory(self, *, user_id: str) -> list[ProceduralMemoryRecord]:
        return [record for record in self.procedural if record.user_id == user_id]

    async def delete_procedural_memory(self, *, user_id: str, key: str) -> bool:
        before = len(self.procedural)
        self.procedural = [
            record
            for record in self.procedural
            if not (record.user_id == user_id and record.key == key)
        ]
        return len(self.procedural) < before

    async def upsert_session_stats(self, record: SessionStatsRecord) -> SessionStatsRecord:
        self.sessions[record.session_id] = record
        return record

    async def get_session_stats(self, *, session_id: str) -> SessionStatsRecord | None:
        return self.sessions.get(session_id)


class StubSummarizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def summarize(
        self,
        *,
        user_id: str,
        session_id: str,
        chunks: Sequence[MemoryChunk],
    ) -> str:
        self.calls.append(
            {
                "user_id": user_id,
                "session_id": session_id,
                "chunk_ids": [chunk.id for chunk in chunks],
            }
        )
        return f"Summary for {session_id}."


class StubEmbedder:
    async def encode_batch(self, chunks: list[str]) -> list[list[float]]:
        return [[float(len(chunks[0].split()))]]


def _chunk(
    *,
    memory_id: str,
    session_id: str,
    created_at: datetime,
    compressed: bool = False,
) -> MemoryChunk:
    return MemoryChunk(
        id=memory_id,
        user_id="user_a",
        session_id=session_id,
        memory_type=MemoryType.EPISODIC,
        content=f"event for {session_id}",
        importance=0.8,
        token_count=3,
        embedding=[0.1, 0.2],
        compressed=compressed,
        created_at=created_at,
        updated_at=created_at,
    )


@pytest.mark.asyncio
async def test_compress_user_skips_when_threshold_not_exceeded() -> None:
    now = datetime(2026, 4, 19, 12, 0, tzinfo=UTC)
    storage = FakeStorage()
    storage.chunks["mem_1"] = _chunk(
        memory_id="mem_1",
        session_id="sess_1",
        created_at=now - timedelta(days=2),
    )
    storage.chunks["mem_2"] = _chunk(
        memory_id="mem_2",
        session_id="sess_2",
        created_at=now - timedelta(days=1),
    )

    compressor = MemoryCompressor(storage=storage, compression_threshold=2)
    result = await compressor.compress_user("user_a")

    assert result.total_uncompressed_sessions == 2
    assert result.sessions_compressed == 0
    assert result.summaries_created == 0
    assert all(not chunk.compressed for chunk in storage.chunks.values())


@pytest.mark.asyncio
async def test_compress_user_creates_summary_and_marks_oldest_session() -> None:
    now = datetime(2026, 4, 19, 12, 0, tzinfo=UTC)
    storage = FakeStorage()
    storage.chunks["mem_1"] = _chunk(
        memory_id="mem_1",
        session_id="sess_oldest",
        created_at=now - timedelta(days=3),
    )
    storage.chunks["mem_2"] = _chunk(
        memory_id="mem_2",
        session_id="sess_middle",
        created_at=now - timedelta(days=2),
    )
    storage.chunks["mem_3"] = _chunk(
        memory_id="mem_3",
        session_id="sess_latest",
        created_at=now - timedelta(days=1),
    )

    storage.sessions["sess_oldest"] = SessionStatsRecord(
        session_id="sess_oldest",
        user_id="user_a",
        memory_count=1,
        total_tokens_stored=3,
        started_at=now - timedelta(days=3),
        last_activity=now - timedelta(days=3),
    )

    summarizer = StubSummarizer()
    compressor = MemoryCompressor(
        storage=storage,
        summarizer=summarizer,
        embedder=StubEmbedder(),
        compression_threshold=2,
        sessions_to_compress=1,
        now_provider=lambda: now,
    )

    result = await compressor.compress_user("user_a")

    assert result.total_uncompressed_sessions == 3
    assert result.sessions_compressed == 1
    assert result.summaries_created == 1
    assert result.memories_marked_compressed == 1
    assert result.compressed_session_ids == ("sess_oldest",)

    oldest_chunk = storage.chunks["mem_1"]
    assert oldest_chunk.compressed is True

    summary_chunks = [
        chunk for chunk in storage.chunks.values() if chunk.compression_source
    ]
    assert len(summary_chunks) == 1
    summary = summary_chunks[0]
    assert summary.memory_type is MemoryType.SEMANTIC
    assert summary.source_session_id == "sess_oldest"
    assert summary.embedding == [3.0]

    assert storage.sessions["sess_oldest"].compressed is True
    assert summarizer.calls[0]["session_id"] == "sess_oldest"


@pytest.mark.asyncio
async def test_compress_user_force_ignores_threshold() -> None:
    now = datetime(2026, 4, 19, 12, 0, tzinfo=UTC)
    storage = FakeStorage()
    storage.chunks["mem_1"] = _chunk(
        memory_id="mem_1",
        session_id="sess_1",
        created_at=now - timedelta(days=1),
    )

    compressor = MemoryCompressor(
        storage=storage,
        summarizer=StubSummarizer(),
        compression_threshold=10,
        now_provider=lambda: now,
    )

    result = await compressor.compress_user("user_a", force=True, sessions_to_compress=1)

    assert result.sessions_compressed == 1
    assert result.summaries_created == 1


@pytest.mark.asyncio
async def test_compress_user_wraps_unexpected_errors() -> None:
    storage = FakeStorage()
    storage.raise_on_list = True
    compressor = MemoryCompressor(storage=storage)

    with pytest.raises(CompressionError):
        await compressor.compress_user("user_a")


@pytest.mark.asyncio
async def test_heuristic_summarizer_limits_sentence_count() -> None:
    summarizer = HeuristicSessionSummarizer(max_sentences=3)
    base = datetime(2026, 4, 19, 10, 0, tzinfo=UTC)
    chunks = [
        MemoryChunk(
            id="mem_1",
            user_id="user_a",
            session_id="sess_1",
            memory_type=MemoryType.EPISODIC,
            content=(
                "First sentence. Second sentence! Third sentence? Fourth sentence remains."
            ),
            importance=0.7,
            token_count=8,
            embedding=[0.1],
            created_at=base,
            updated_at=base,
        )
    ]

    summary = await summarizer.summarize(
        user_id="user_a",
        session_id="sess_1",
        chunks=chunks,
    )

    assert "Fourth sentence remains" not in summary
    assert summary.count("sentence") == 3
