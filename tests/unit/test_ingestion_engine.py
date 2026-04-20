from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from memory_vault.exceptions import IngestionError, StorageError
from memory_vault.ingestion.engine import IngestionEngine
from memory_vault.models import MemoryChunk, MemoryType, PaginatedResult
from memory_vault.storage.base import (
    MemoryListQuery,
    MemorySearchQuery,
    ProceduralMemoryRecord,
    SessionStatsRecord,
    StorageBackend,
)


class StubChunker:
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    def chunk(self, text: str) -> list[str]:
        del text
        return list(self._chunks)


class StubEmbedder:
    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = vectors

    async def encode_batch(self, chunks: list[str]) -> list[list[float]]:
        if len(chunks) != len(self._vectors):
            raise RuntimeError("vector count mismatch in test stub")
        return [vector[:] for vector in self._vectors]


class StubScorer:
    def __init__(self, scores: list[float], threshold: float = 0.3) -> None:
        self._scores = scores
        self.threshold = threshold
        self.received_existing_embeddings: list[list[list[float]]] = []

    def score(
        self,
        chunk_text: str,
        *,
        chunk_embedding: list[float],
        existing_embeddings: list[list[float]] | None = None,
    ) -> float:
        del chunk_text, chunk_embedding
        snapshot = [vector[:] for vector in (existing_embeddings or [])]
        self.received_existing_embeddings.append(snapshot)
        if not self._scores:
            return 1.0
        return self._scores.pop(0)


class FakeStorage(StorageBackend):
    def __init__(self) -> None:
        self.stored_chunks: dict[str, MemoryChunk] = {}
        self.procedural_records: list[ProceduralMemoryRecord] = []
        self.session_records: dict[str, SessionStatsRecord] = {}
        self.query_results: list[MemoryChunk] = []
        self.fail_on_upsert = False

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
        del query
        return [chunk.model_copy() for chunk in self.query_results]

    async def delete_vectors(self, memory_ids: Sequence[str], *, user_id: str) -> int:
        del memory_ids, user_id
        return 0

    async def delete_vectors_for_user(self, *, user_id: str) -> int:
        del user_id
        return 0

    async def upsert_memory_chunks(self, chunks: Sequence[MemoryChunk]) -> list[MemoryChunk]:
        if self.fail_on_upsert:
            raise StorageError("simulated failure")

        for chunk in chunks:
            self.stored_chunks[chunk.id] = chunk
        return list(chunks)

    async def get_memory_chunk(self, *, memory_id: str, user_id: str) -> MemoryChunk | None:
        chunk = self.stored_chunks.get(memory_id)
        if chunk is None or chunk.user_id != user_id:
            return None
        return chunk

    async def list_memory_chunks(self, query: MemoryListQuery) -> PaginatedResult[MemoryChunk]:
        del query
        items = list(self.stored_chunks.values())
        return PaginatedResult[MemoryChunk](
            items=items,
            total=len(items),
            page=1,
            page_size=20,
        )

    async def delete_memory_chunk(self, *, memory_id: str, user_id: str) -> bool:
        chunk = self.stored_chunks.get(memory_id)
        if chunk is None or chunk.user_id != user_id:
            return False
        self.stored_chunks.pop(memory_id)
        return True

    async def delete_memory_chunks_for_user(self, *, user_id: str) -> int:
        to_delete = [
            memory_id
            for memory_id, chunk in self.stored_chunks.items()
            if chunk.user_id == user_id
        ]
        for memory_id in to_delete:
            self.stored_chunks.pop(memory_id)
        return len(to_delete)

    async def upsert_procedural_memory(
        self,
        record: ProceduralMemoryRecord,
    ) -> ProceduralMemoryRecord:
        self.procedural_records.append(record)
        return record

    async def list_procedural_memory(self, *, user_id: str) -> list[ProceduralMemoryRecord]:
        return [record for record in self.procedural_records if record.user_id == user_id]

    async def delete_procedural_memory(self, *, user_id: str, key: str) -> bool:
        initial_count = len(self.procedural_records)
        self.procedural_records = [
            record
            for record in self.procedural_records
            if not (record.user_id == user_id and record.key == key)
        ]
        return len(self.procedural_records) < initial_count

    async def upsert_session_stats(self, record: SessionStatsRecord) -> SessionStatsRecord:
        self.session_records[record.session_id] = record
        return record

    async def get_session_stats(self, *, session_id: str) -> SessionStatsRecord | None:
        return self.session_records.get(session_id)


def _timestamp() -> datetime:
    return datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def _existing_chunk(*, memory_id: str, embedding: list[float]) -> MemoryChunk:
    return MemoryChunk(
        id=memory_id,
        user_id="user_a",
        session_id="sess_old",
        memory_type=MemoryType.SEMANTIC,
        content="older fact",
        importance=0.8,
        token_count=2,
        embedding=embedding,
    )


@pytest.mark.asyncio
async def test_ingest_returns_empty_when_chunker_emits_no_chunks() -> None:
    storage = FakeStorage()
    engine = IngestionEngine(
        storage=storage,
        chunker=StubChunker([]),
        embedder=StubEmbedder([]),
        scorer=StubScorer([1.0]),
    )

    saved = await engine.ingest("some text", user_id="user_a", session_id="sess_a")

    assert saved == []
    assert storage.stored_chunks == {}
    assert storage.session_records == {}


@pytest.mark.asyncio
async def test_ingest_discards_below_threshold_and_persists_accepted_chunks() -> None:
    storage = FakeStorage()
    engine = IngestionEngine(
        storage=storage,
        chunker=StubChunker(["keep this", "drop this"]),
        embedder=StubEmbedder([[1.0, 0.0], [0.0, 1.0]]),
        scorer=StubScorer([0.9, 0.1], threshold=0.3),
        token_counter=lambda text: len(text.split()),
        now_provider=_timestamp,
    )

    saved = await engine.ingest(
        "ignored",
        user_id="user_a",
        session_id="sess_a",
        memory_type_hint=MemoryType.SEMANTIC,
    )

    assert len(saved) == 1
    assert saved[0].memory_type is MemoryType.SEMANTIC
    assert saved[0].token_count == 2
    assert saved[0].id.startswith("mem_")

    stats = storage.session_records["sess_a"]
    assert stats.memory_count == 1
    assert stats.total_tokens_stored == 2


@pytest.mark.asyncio
async def test_ingest_routes_procedural_and_writes_procedural_record() -> None:
    storage = FakeStorage()
    engine = IngestionEngine(
        storage=storage,
        chunker=StubChunker(["I prefer concise technical answers."]),
        embedder=StubEmbedder([[0.2, 0.8]]),
        scorer=StubScorer([0.75]),
        now_provider=_timestamp,
    )

    saved = await engine.ingest(
        "ignored",
        user_id="user_a",
        session_id="sess_a",
    )

    assert len(saved) == 1
    assert saved[0].memory_type is MemoryType.PROCEDURAL
    assert len(storage.procedural_records) == 1

    record = storage.procedural_records[0]
    assert record.user_id == "user_a"
    assert record.source_chunk_id == saved[0].id
    assert record.value == "I prefer concise technical answers."
    assert record.key.startswith("preference_")


@pytest.mark.asyncio
async def test_ingest_memory_type_hint_overrides_auto_routing() -> None:
    storage = FakeStorage()
    engine = IngestionEngine(
        storage=storage,
        chunker=StubChunker(["I prefer concise technical answers."]),
        embedder=StubEmbedder([[0.2, 0.8]]),
        scorer=StubScorer([0.75]),
    )

    saved = await engine.ingest(
        "ignored",
        user_id="user_a",
        session_id="sess_a",
        memory_type_hint=MemoryType.EPISODIC,
    )

    assert len(saved) == 1
    assert saved[0].memory_type is MemoryType.EPISODIC
    assert storage.procedural_records == []


@pytest.mark.asyncio
async def test_ingest_uses_existing_embeddings_for_novelty_inputs() -> None:
    storage = FakeStorage()
    storage.query_results = [
        _existing_chunk(memory_id="mem_existing", embedding=[0.5, 0.5])
    ]
    scorer = StubScorer([0.9])
    engine = IngestionEngine(
        storage=storage,
        chunker=StubChunker(["new chunk"]),
        embedder=StubEmbedder([[1.0, 0.0]]),
        scorer=scorer,
    )

    await engine.ingest("ignored", user_id="user_a", session_id="sess_a")

    assert len(scorer.received_existing_embeddings) == 1
    assert scorer.received_existing_embeddings[0] == [[0.5, 0.5]]


@pytest.mark.asyncio
async def test_ingest_updates_existing_session_stats_totals() -> None:
    storage = FakeStorage()
    storage.session_records["sess_a"] = SessionStatsRecord(
        session_id="sess_a",
        user_id="user_a",
        memory_count=2,
        total_tokens_stored=10,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        last_activity=datetime(2026, 1, 1, tzinfo=UTC),
    )
    engine = IngestionEngine(
        storage=storage,
        chunker=StubChunker(["one two three"]),
        embedder=StubEmbedder([[0.3, 0.7]]),
        scorer=StubScorer([0.8]),
        token_counter=lambda text: len(text.split()),
        now_provider=_timestamp,
    )

    await engine.ingest("ignored", user_id="user_a", session_id="sess_a")

    stats = storage.session_records["sess_a"]
    assert stats.memory_count == 3
    assert stats.total_tokens_stored == 13
    assert stats.started_at == datetime(2026, 1, 1, tzinfo=UTC)
    assert stats.last_activity == _timestamp()


@pytest.mark.asyncio
async def test_ingest_wraps_unexpected_storage_failure() -> None:
    storage = FakeStorage()
    storage.fail_on_upsert = True
    engine = IngestionEngine(
        storage=storage,
        chunker=StubChunker(["persist me"]),
        embedder=StubEmbedder([[0.1, 0.2]]),
        scorer=StubScorer([0.9]),
    )

    with pytest.raises(IngestionError):
        await engine.ingest("ignored", user_id="user_a", session_id="sess_a")


@pytest.mark.asyncio
async def test_ingest_validates_required_identifiers() -> None:
    engine = IngestionEngine(
        storage=FakeStorage(),
        chunker=StubChunker(["x"]),
        embedder=StubEmbedder([[0.1]]),
        scorer=StubScorer([0.9]),
    )

    with pytest.raises(ValueError):
        await engine.ingest("text", user_id="", session_id="sess_a")
    with pytest.raises(ValueError):
        await engine.ingest("text", user_id="user_a", session_id="")


@pytest.mark.asyncio
async def test_ingest_does_not_overwrite_foreign_session_stats() -> None:
    storage = FakeStorage()
    storage.session_records["sess_shared"] = SessionStatsRecord(
        session_id="sess_shared",
        user_id="user_other",
        memory_count=4,
        total_tokens_stored=20,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        last_activity=datetime(2026, 1, 1, tzinfo=UTC),
    )
    engine = IngestionEngine(
        storage=storage,
        chunker=StubChunker(["new user chunk"]),
        embedder=StubEmbedder([[0.1, 0.9]]),
        scorer=StubScorer([0.8]),
    )

    saved = await engine.ingest("ignored", user_id="user_a", session_id="sess_shared")

    assert len(saved) == 1
    assert storage.session_records["sess_shared"].user_id == "user_other"
    assert storage.session_records["sess_shared"].memory_count == 4
