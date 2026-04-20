from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, ClassVar

import pytest
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
        self.sessions: dict[str, SessionStatsRecord] = {}

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
        self.sessions[record.session_id] = record
        return record

    async def get_session_stats(self, *, session_id: str) -> SessionStatsRecord | None:
        return self.sessions.get(session_id)


class FakeMemoryCompressor:
    init_calls: ClassVar[list[dict[str, Any]]] = []
    compress_calls: ClassVar[list[str]] = []

    def __init__(
        self,
        *,
        storage: StorageBackend,
        compression_threshold: int,
    ) -> None:
        self.storage = storage
        self.compression_threshold = compression_threshold
        FakeMemoryCompressor.init_calls.append(
            {
                "storage": storage,
                "compression_threshold": compression_threshold,
            }
        )

    async def compress_session(self, session_id: str) -> None:
        FakeMemoryCompressor.compress_calls.append(session_id)


def _reset_fake_compressor() -> None:
    FakeMemoryCompressor.init_calls.clear()
    FakeMemoryCompressor.compress_calls.clear()


def _session_record(*, session_id: str) -> SessionStatsRecord:
    now = datetime(2026, 4, 19, 12, 0, tzinfo=UTC)
    return SessionStatsRecord(
        session_id=session_id,
        user_id="user_123",
        memory_count=12,
        total_tokens_stored=3200,
        started_at=now,
        last_activity=now,
        compressed=False,
    )


def test_get_session_stats_returns_record() -> None:
    storage = FakeStorage()
    storage.sessions["sess_xyz"] = _session_record(session_id="sess_xyz")

    settings = Settings(storage_backend="chroma", metadata_backend="sqlite")
    app = create_app(settings=settings, storage=storage)

    with TestClient(app) as client:
        response = client.get("/v1/session/sess_xyz/stats?user_id=user_123")

    assert response.status_code == 200
    assert response.json() == {
        "session_id": "sess_xyz",
        "user_id": "user_123",
        "memory_count": 12,
        "total_tokens_stored": 3200,
        "started_at": "2026-04-19T12:00:00Z",
        "last_activity": "2026-04-19T12:00:00Z",
        "ended_at": None,
        "compressed": False,
    }


def test_get_session_stats_returns_404_when_missing() -> None:
    settings = Settings(storage_backend="chroma", metadata_backend="sqlite")
    app = create_app(settings=settings, storage=FakeStorage())

    with TestClient(app) as client:
        response = client.get("/v1/session/sess_missing/stats?user_id=user_123")

    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found"}


def test_post_session_compress_queues_background_job(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _reset_fake_compressor()
    monkeypatch.setattr("memory_vault.api.routes.session.MemoryCompressor", FakeMemoryCompressor)

    storage = FakeStorage()
    storage.sessions["sess_xyz"] = _session_record(session_id="sess_xyz")

    settings = Settings(
        storage_backend="chroma",
        metadata_backend="sqlite",
        compression_threshold=7,
    )
    app = create_app(settings=settings, storage=storage)

    with TestClient(app) as client:
        response = client.post("/v1/session/sess_xyz/compress?user_id=user_123")

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["job_id"].startswith("job_")
    assert body["message"] == "Compression queued. Check /v1/jobs/{job_id} for status."

    assert FakeMemoryCompressor.init_calls[0]["compression_threshold"] == 7
    assert FakeMemoryCompressor.compress_calls == ["sess_xyz"]


def test_post_session_compress_returns_404_when_missing(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _reset_fake_compressor()
    monkeypatch.setattr("memory_vault.api.routes.session.MemoryCompressor", FakeMemoryCompressor)

    settings = Settings(storage_backend="chroma", metadata_backend="sqlite")
    app = create_app(settings=settings, storage=FakeStorage())

    with TestClient(app) as client:
        response = client.post("/v1/session/sess_missing/compress?user_id=user_123")

    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found"}
    assert FakeMemoryCompressor.compress_calls == []


def test_get_session_stats_returns_404_for_wrong_user_scope() -> None:
    storage = FakeStorage()
    storage.sessions["sess_xyz"] = _session_record(session_id="sess_xyz")

    settings = Settings(storage_backend="chroma", metadata_backend="sqlite")
    app = create_app(settings=settings, storage=storage)

    with TestClient(app) as client:
        response = client.get("/v1/session/sess_xyz/stats?user_id=user_other")

    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found"}


def test_post_session_compress_returns_404_for_wrong_user_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_fake_compressor()
    monkeypatch.setattr("memory_vault.api.routes.session.MemoryCompressor", FakeMemoryCompressor)

    storage = FakeStorage()
    storage.sessions["sess_xyz"] = _session_record(session_id="sess_xyz")

    settings = Settings(storage_backend="chroma", metadata_backend="sqlite")
    app = create_app(settings=settings, storage=storage)

    with TestClient(app) as client:
        response = client.post("/v1/session/sess_xyz/compress?user_id=user_other")

    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found"}
    assert FakeMemoryCompressor.compress_calls == []
