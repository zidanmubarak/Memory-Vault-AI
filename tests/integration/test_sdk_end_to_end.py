from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from math import sqrt

import pytest

from memory_layer.ingestion.engine import IngestionEngine
from memory_layer.ingestion.scorer import ImportanceScorer
from memory_layer.models import MemoryChunk, MemoryConfig, MemoryType, PaginatedResult
from memory_layer.retrieval.engine import RetrievalEngine
from memory_layer.sdk import MemoryLayer
from memory_layer.storage.base import (
    MemoryListQuery,
    MemorySearchQuery,
    ProceduralMemoryRecord,
    SessionStatsRecord,
    StorageBackend,
)


class SingleChunker:
    def chunk(self, text: str) -> list[str]:
        return [text] if text.strip() else []


class DeterministicEmbedder:
    async def encode_batch(self, chunks: list[str]) -> list[list[float]]:
        return [self._encode(text) for text in chunks]

    @staticmethod
    def _encode(text: str) -> list[float]:
        lowered = text.lower()
        vector = [
            float(lowered.count("fastapi") + lowered.count("api")),
            float(
                lowered.count("postgresql")
                + lowered.count("database")
                + lowered.count("db")
            ),
            float(
                lowered.count("prefer")
                + lowered.count("concise")
                + lowered.count("style")
            ),
        ]
        if vector == [0.0, 0.0, 0.0]:
            return [1.0, 0.0, 0.0]
        return vector


class InMemoryStorage(StorageBackend):
    def __init__(self) -> None:
        self._initialized = False
        self._chunks: dict[str, MemoryChunk] = {}
        self._procedural: list[ProceduralMemoryRecord] = []
        self._sessions: dict[str, SessionStatsRecord] = {}

    async def initialize(self) -> None:
        self._initialized = True

    async def close(self) -> None:
        self._initialized = False

    async def healthcheck(self) -> dict[str, str]:
        return {
            "status": "ok",
            "vector": "ok",
            "metadata": "ok",
        }

    async def upsert_vectors(self, chunks: Sequence[MemoryChunk]) -> None:
        del chunks

    async def query_vectors(self, query: MemorySearchQuery) -> list[MemoryChunk]:
        candidates: list[MemoryChunk] = []

        for chunk in self._chunks.values():
            if chunk.user_id != query.user_id:
                continue
            if chunk.embedding is None:
                continue
            if chunk.importance < query.min_importance:
                continue
            if query.memory_types and chunk.memory_type not in query.memory_types:
                continue
            if chunk.compressed and not query.include_compressed:
                continue

            score = self._cosine_similarity(query.query_embedding, chunk.embedding)
            candidates.append(chunk.model_copy(update={"relevance_score": score}))

        candidates.sort(key=lambda chunk: chunk.relevance_score or 0.0, reverse=True)
        return candidates[: query.top_k]

    async def delete_vectors(self, memory_ids: Sequence[str], *, user_id: str) -> int:
        deleted = 0
        for memory_id in memory_ids:
            chunk = self._chunks.get(memory_id)
            if chunk is not None and chunk.user_id == user_id:
                self._chunks.pop(memory_id)
                deleted += 1
        return deleted

    async def delete_vectors_for_user(self, *, user_id: str) -> int:
        target_ids = [
            memory_id
            for memory_id, chunk in self._chunks.items()
            if chunk.user_id == user_id
        ]
        for memory_id in target_ids:
            self._chunks.pop(memory_id)
        return len(target_ids)

    async def upsert_memory_chunks(self, chunks: Sequence[MemoryChunk]) -> list[MemoryChunk]:
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
        return list(chunks)

    async def get_memory_chunk(self, *, memory_id: str, user_id: str) -> MemoryChunk | None:
        chunk = self._chunks.get(memory_id)
        if chunk is None or chunk.user_id != user_id:
            return None
        return chunk

    async def list_memory_chunks(self, query: MemoryListQuery) -> PaginatedResult[MemoryChunk]:
        items = [
            chunk
            for chunk in self._chunks.values()
            if chunk.user_id == query.user_id
            and (query.memory_type is None or chunk.memory_type is query.memory_type)
            and (query.include_compressed or not chunk.compressed)
        ]
        items.sort(key=lambda chunk: chunk.created_at, reverse=True)
        total = len(items)
        start = (query.page - 1) * query.page_size
        end = start + query.page_size
        return PaginatedResult[MemoryChunk](
            items=items[start:end],
            total=total,
            page=query.page,
            page_size=query.page_size,
        )

    async def delete_memory_chunk(self, *, memory_id: str, user_id: str) -> bool:
        chunk = self._chunks.get(memory_id)
        if chunk is None or chunk.user_id != user_id:
            return False
        self._chunks.pop(memory_id)
        return True

    async def delete_memory_chunks_for_user(self, *, user_id: str) -> int:
        memory_ids = [
            memory_id
            for memory_id, chunk in self._chunks.items()
            if chunk.user_id == user_id
        ]
        for memory_id in memory_ids:
            self._chunks.pop(memory_id)
        return len(memory_ids)

    async def upsert_procedural_memory(
        self,
        record: ProceduralMemoryRecord,
    ) -> ProceduralMemoryRecord:
        filtered = [
            existing
            for existing in self._procedural
            if not (existing.user_id == record.user_id and existing.key == record.key)
        ]
        filtered.append(record)
        self._procedural = filtered
        return record

    async def list_procedural_memory(self, *, user_id: str) -> list[ProceduralMemoryRecord]:
        return [record for record in self._procedural if record.user_id == user_id]

    async def delete_procedural_memory(self, *, user_id: str, key: str) -> bool:
        initial = len(self._procedural)
        self._procedural = [
            record
            for record in self._procedural
            if not (record.user_id == user_id and record.key == key)
        ]
        return len(self._procedural) < initial

    async def upsert_session_stats(self, record: SessionStatsRecord) -> SessionStatsRecord:
        self._sessions[record.session_id] = record
        return record

    async def get_session_stats(self, *, session_id: str) -> SessionStatsRecord | None:
        return self._sessions.get(session_id)

    @staticmethod
    def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        if len(a) != len(b):
            return 0.0

        dot = sum(left * right for left, right in zip(a, b, strict=True))
        norm_a = sqrt(sum(value * value for value in a))
        norm_b = sqrt(sum(value * value for value in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)


@pytest.mark.asyncio
async def test_memory_layer_save_recall_end_to_end() -> None:
    now = datetime(2026, 4, 19, 12, 0, tzinfo=UTC)
    storage = InMemoryStorage()
    embedder = DeterministicEmbedder()
    ingestion = IngestionEngine(
        storage=storage,
        chunker=SingleChunker(),
        embedder=embedder,
        scorer=ImportanceScorer(threshold=0.0),
        token_counter=lambda text: len(text.split()),
        now_provider=lambda: now,
    )
    retrieval = RetrievalEngine(
        storage=storage,
        embedder=embedder,
        reranker_enabled=False,
        token_counter=lambda text: len(text.split()),
    )

    sdk = MemoryLayer(
        user_id="user_e2e",
        session_id="sess_e2e",
        config=MemoryConfig(token_budget=64, top_k=4, importance_threshold=0.0),
        storage=storage,
        ingestion_engine=ingestion,
        retrieval_engine=retrieval,
    )

    await sdk.save("I am building APIs with FastAPI and PostgreSQL.")
    await sdk.save("I prefer concise answers with practical examples.")

    recalled = await sdk.recall(
        "What stack is the user using?",
        top_k=3,
        token_budget=64,
        memory_types=[MemoryType.SEMANTIC, MemoryType.PROCEDURAL],
    )

    assert len(recalled.memories) >= 2
    assert any(chunk.memory_type is MemoryType.SEMANTIC for chunk in recalled.memories)
    assert any(chunk.memory_type is MemoryType.PROCEDURAL for chunk in recalled.memories)
    assert recalled.prompt_block.startswith("<memory>\n")

    listed = await sdk.list(page=1, page_size=10)
    assert listed.total == 2

    deleted_one = await sdk.forget(memory_id=listed.items[0].id)
    assert deleted_one is True

    deleted_remaining = await sdk.forget_all(confirm=True)
    assert deleted_remaining >= 1

    await sdk.close()
