from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, ClassVar

from fastapi.testclient import TestClient

from memory_vault.api.main import create_app
from memory_vault.config import Settings
from memory_vault.models import MemoryChunk, PaginatedResult
from memory_vault.storage.base import (
    MemoryListQuery,
    MemorySearchQuery,
    ProceduralMemoryRecord,
    SessionStatsRecord,
    StorageBackend,
)


class FakeStorage(StorageBackend):
    def __init__(self) -> None:
        self.initialized = False
        self.closed = False

    async def initialize(self) -> None:
        self.initialized = True

    async def close(self) -> None:
        self.closed = True

    async def healthcheck(self) -> dict[str, str]:
        return {"status": "ok", "vector": "ok", "metadata": "ok"}

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


class FakeMemoryLayer:
    created: ClassVar[list[dict[str, Any]]] = []
    list_calls: ClassVar[list[dict[str, Any]]] = []
    upsert_calls: ClassVar[list[dict[str, Any]]] = []
    delete_calls: ClassVar[list[dict[str, Any]]] = []

    def __init__(
        self,
        *,
        user_id: str,
        session_id: str | None = None,
        config: Any | None = None,
        storage: StorageBackend | None = None,
    ) -> None:
        del session_id
        self.user_id = user_id
        self.config = config
        self.storage = storage
        FakeMemoryLayer.created.append(
            {
                "user_id": user_id,
                "config": config,
                "storage": storage,
            }
        )

    async def list_procedural_memory(self) -> list[ProceduralMemoryRecord]:
        FakeMemoryLayer.list_calls.append({"user_id": self.user_id})
        now = datetime(2026, 4, 19, 13, 0, tzinfo=UTC)
        return [
            ProceduralMemoryRecord(
                user_id=self.user_id,
                key="tone",
                value="Use concise technical responses.",
                confidence=0.91,
                updated_at=now,
                source_chunk_id="mem_source_1",
            )
        ]

    async def upsert_procedural_memory(
        self,
        *,
        key: str,
        value: str,
        confidence: float = 1.0,
        source_chunk_id: str | None = None,
    ) -> ProceduralMemoryRecord:
        FakeMemoryLayer.upsert_calls.append(
            {
                "user_id": self.user_id,
                "key": key,
                "value": value,
                "confidence": confidence,
                "source_chunk_id": source_chunk_id,
            }
        )
        now = datetime(2026, 4, 19, 13, 5, tzinfo=UTC)
        return ProceduralMemoryRecord(
            user_id=self.user_id,
            key=key,
            value=value,
            confidence=confidence,
            updated_at=now,
            source_chunk_id=source_chunk_id,
        )

    async def delete_procedural_memory(self, *, key: str) -> bool:
        FakeMemoryLayer.delete_calls.append({"user_id": self.user_id, "key": key})
        return key != "missing"


def _clear_fake_memory_vault() -> None:
    FakeMemoryLayer.created.clear()
    FakeMemoryLayer.list_calls.clear()
    FakeMemoryLayer.upsert_calls.clear()
    FakeMemoryLayer.delete_calls.clear()


def test_get_procedural_memory_returns_items(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _clear_fake_memory_vault()
    monkeypatch.setattr("memory_vault.api.routes.procedural.MemoryLayer", FakeMemoryLayer)

    settings = Settings(storage_backend="chroma", metadata_backend="sqlite")
    app = create_app(settings=settings, storage=FakeStorage())

    with TestClient(app) as client:
        response = client.get("/v1/procedural", params={"user_id": "user_123"})

    assert response.status_code == 200
    assert response.json() == {
        "items": [
            {
                "key": "tone",
                "value": "Use concise technical responses.",
                "confidence": 0.91,
                "updated_at": "2026-04-19T13:00:00Z",
                "source_chunk_id": "mem_source_1",
            }
        ]
    }
    assert FakeMemoryLayer.list_calls == [{"user_id": "user_123"}]


def test_put_procedural_memory_upserts_item(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _clear_fake_memory_vault()
    monkeypatch.setattr("memory_vault.api.routes.procedural.MemoryLayer", FakeMemoryLayer)

    settings = Settings(storage_backend="chroma", metadata_backend="sqlite")
    app = create_app(settings=settings, storage=FakeStorage())

    payload = {
        "user_id": "user_123",
        "key": "style",
        "value": "Prefer direct and actionable answers.",
        "confidence": 0.88,
        "source_chunk_id": "mem_source_2",
    }

    with TestClient(app) as client:
        response = client.put("/v1/procedural", json=payload)

    assert response.status_code == 200
    assert response.json() == {
        "key": "style",
        "value": "Prefer direct and actionable answers.",
        "confidence": 0.88,
        "updated_at": "2026-04-19T13:05:00Z",
        "source_chunk_id": "mem_source_2",
    }
    assert FakeMemoryLayer.upsert_calls == [
        {
            "user_id": "user_123",
            "key": "style",
            "value": "Prefer direct and actionable answers.",
            "confidence": 0.88,
            "source_chunk_id": "mem_source_2",
        }
    ]


def test_delete_procedural_memory_returns_deleted(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _clear_fake_memory_vault()
    monkeypatch.setattr("memory_vault.api.routes.procedural.MemoryLayer", FakeMemoryLayer)

    settings = Settings(storage_backend="chroma", metadata_backend="sqlite")
    app = create_app(settings=settings, storage=FakeStorage())

    with TestClient(app) as client:
        response = client.delete("/v1/procedural/style", params={"user_id": "user_123"})

    assert response.status_code == 200
    assert response.json() == {"deleted": True, "key": "style"}
    assert FakeMemoryLayer.delete_calls == [{"user_id": "user_123", "key": "style"}]


def test_delete_procedural_memory_returns_404_when_missing(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _clear_fake_memory_vault()
    monkeypatch.setattr("memory_vault.api.routes.procedural.MemoryLayer", FakeMemoryLayer)

    settings = Settings(storage_backend="chroma", metadata_backend="sqlite")
    app = create_app(settings=settings, storage=FakeStorage())

    with TestClient(app) as client:
        response = client.delete("/v1/procedural/missing", params={"user_id": "user_123"})

    assert response.status_code == 404
    assert response.json() == {"detail": "Procedural memory not found"}
