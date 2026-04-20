from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from memory_vault.ingestion.engine import IngestionEngine
from memory_vault.models import MemoryChunk, MemoryType, PaginatedResult
from memory_vault.plugins import MemoryTypePlugin, MemoryTypePluginRegistry
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

    def score(
        self,
        chunk_text: str,
        *,
        chunk_embedding: list[float],
        existing_embeddings: list[list[float]] | None = None,
    ) -> float:
        del chunk_text, chunk_embedding, existing_embeddings
        if not self._scores:
            return 1.0
        return self._scores.pop(0)


class FakeStorage(StorageBackend):
    def __init__(self) -> None:
        self.stored_chunks: dict[str, MemoryChunk] = {}
        self.procedural_records: list[ProceduralMemoryRecord] = []
        self.session_records: dict[str, SessionStatsRecord] = {}

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


class _ProjectProfilePlugin(MemoryTypePlugin):
    name = "project_profile"
    base_memory_type = MemoryType.SEMANTIC
    priority = 200

    def matches(self, chunk_text: str) -> bool:
        return "project profile" in chunk_text.lower()

    def metadata(self, chunk_text: str) -> dict[str, object]:
        del chunk_text
        return {"source": "plugin_test", "confidence_bucket": "high"}


@pytest.mark.asyncio
async def test_ingestion_plugin_assigns_custom_memory_type_metadata() -> None:
    storage = FakeStorage()
    registry = MemoryTypePluginRegistry()
    registry.register(_ProjectProfilePlugin())

    engine = IngestionEngine(
        storage=storage,
        chunker=StubChunker(["Project profile: backend APIs and data model summary."]),
        embedder=StubEmbedder([[0.9, 0.1]]),
        scorer=StubScorer([0.8]),
        token_counter=lambda text: len(text.split()),
        now_provider=lambda: datetime(2026, 4, 19, 12, 0, tzinfo=UTC),
        plugin_registry=registry,
    )

    saved = await engine.ingest("ignored", user_id="user_a", session_id="sess_a")

    assert len(saved) == 1
    assert saved[0].memory_type is MemoryType.SEMANTIC
    assert saved[0].metadata["custom_memory_type"] == "project_profile"
    assert saved[0].metadata["source"] == "plugin_test"


@pytest.mark.asyncio
async def test_ingestion_hint_bypasses_plugin_classification() -> None:
    storage = FakeStorage()
    registry = MemoryTypePluginRegistry()
    registry.register(_ProjectProfilePlugin())

    engine = IngestionEngine(
        storage=storage,
        chunker=StubChunker(["Project profile: use plugin match terms."]),
        embedder=StubEmbedder([[0.8, 0.2]]),
        scorer=StubScorer([0.8]),
        now_provider=lambda: datetime(2026, 4, 19, 12, 30, tzinfo=UTC),
        plugin_registry=registry,
    )

    saved = await engine.ingest(
        "ignored",
        user_id="user_a",
        session_id="sess_a",
        memory_type_hint=MemoryType.EPISODIC,
    )

    assert len(saved) == 1
    assert saved[0].memory_type is MemoryType.EPISODIC
    assert "custom_memory_type" not in saved[0].metadata
