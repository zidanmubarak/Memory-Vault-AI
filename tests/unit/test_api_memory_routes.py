from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, ClassVar

from fastapi.testclient import TestClient

from memory_layer.api.main import create_app
from memory_layer.config import Settings
from memory_layer.models import MemoryChunk, MemoryType, PaginatedResult, RecallResult
from memory_layer.storage.base import (
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
    records: ClassVar[list[MemoryChunk]] = []
    save_calls: ClassVar[list[dict[str, Any]]] = []
    recall_calls: ClassVar[list[dict[str, Any]]] = []
    list_calls: ClassVar[list[dict[str, Any]]] = []
    forget_calls: ClassVar[list[dict[str, Any]]] = []
    forget_all_calls: ClassVar[list[dict[str, Any]]] = []

    def __init__(
        self,
        *,
        user_id: str,
        session_id: str | None = None,
        config: Any | None = None,
        storage: StorageBackend | None = None,
    ) -> None:
        self.user_id = user_id
        self.session_id = session_id
        self.config = config
        self.storage = storage
        FakeMemoryLayer.created.append(
            {
                "user_id": user_id,
                "session_id": session_id,
                "config": config,
                "storage": storage,
            }
        )

    async def save(
        self,
        text: str,
        *,
        memory_type_hint: MemoryType | None = None,
        session_id: str | None = None,
    ) -> list[MemoryChunk]:
        FakeMemoryLayer.save_calls.append(
            {
                "text": text,
                "memory_type_hint": memory_type_hint,
                "session_id": session_id,
            }
        )
        now = datetime(2026, 4, 19, 10, 0, tzinfo=UTC)
        chunk = MemoryChunk(
            id=f"mem_{len(FakeMemoryLayer.records) + 1}",
            user_id=self.user_id,
            session_id=session_id or self.session_id or "sess_default",
            memory_type=memory_type_hint or MemoryType.EPISODIC,
            content=text,
            importance=0.72,
            token_count=34,
            embedding=[0.1, 0.2],
            created_at=now,
            updated_at=now,
        )
        FakeMemoryLayer.records.append(chunk)
        return [chunk]

    async def recall(
        self,
        query: str,
        *,
        top_k: int = 5,
        token_budget: int = 2000,
        memory_types: list[MemoryType] | None = None,
    ) -> RecallResult:
        FakeMemoryLayer.recall_calls.append(
            {
                "query": query,
                "top_k": top_k,
                "token_budget": token_budget,
                "memory_types": list(memory_types or []),
            }
        )
        now = datetime(2026, 4, 19, 10, 0, tzinfo=UTC)
        chunk = MemoryChunk(
            id="mem_recall123456",
            user_id=self.user_id,
            session_id="sess_xyz",
            memory_type=MemoryType.SEMANTIC,
            content="User is a backend engineer...",
            importance=0.89,
            token_count=24,
            relevance_score=0.91,
            embedding=[0.3, 0.4],
            created_at=now,
            updated_at=now,
        )
        return RecallResult(
            memories=[chunk],
            total_tokens=24,
            budget_used=0.12,
            prompt_block="<memory>\n[SEMANTIC] User is a backend engineer...\n</memory>",
        )

    async def list(
        self,
        *,
        memory_type: MemoryType | None = None,
        page: int = 1,
        page_size: int = 20,
        include_compressed: bool = False,
    ) -> PaginatedResult[MemoryChunk]:
        FakeMemoryLayer.list_calls.append(
            {
                "memory_type": memory_type,
                "page": page,
                "page_size": page_size,
                "include_compressed": include_compressed,
            }
        )

        filtered = [chunk for chunk in FakeMemoryLayer.records if chunk.user_id == self.user_id]
        if memory_type is not None:
            filtered = [chunk for chunk in filtered if chunk.memory_type is memory_type]
        if not include_compressed:
            filtered = [chunk for chunk in filtered if not chunk.compressed]

        start = (page - 1) * page_size
        end = start + page_size
        return PaginatedResult[MemoryChunk](
            items=filtered[start:end],
            total=len(filtered),
            page=page,
            page_size=page_size,
        )

    async def forget(self, *, memory_id: str) -> bool:
        FakeMemoryLayer.forget_calls.append({"memory_id": memory_id})
        for index, chunk in enumerate(FakeMemoryLayer.records):
            if chunk.user_id == self.user_id and chunk.id == memory_id:
                FakeMemoryLayer.records.pop(index)
                return True
        return False

    async def forget_all(self, *, confirm: bool = False) -> int:
        FakeMemoryLayer.forget_all_calls.append({"confirm": confirm})
        if not confirm:
            raise ValueError("confirm must be true")

        before = len(FakeMemoryLayer.records)
        FakeMemoryLayer.records = [
            chunk for chunk in FakeMemoryLayer.records if chunk.user_id != self.user_id
        ]
        return before - len(FakeMemoryLayer.records)


def _clear_fake_memory_layer() -> None:
    FakeMemoryLayer.created.clear()
    FakeMemoryLayer.records.clear()
    FakeMemoryLayer.save_calls.clear()
    FakeMemoryLayer.recall_calls.clear()
    FakeMemoryLayer.list_calls.clear()
    FakeMemoryLayer.forget_calls.clear()
    FakeMemoryLayer.forget_all_calls.clear()


def test_post_memory_returns_save_result(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _clear_fake_memory_layer()
    monkeypatch.setattr("memory_layer.api.routes.memory.MemoryLayer", FakeMemoryLayer)

    settings = Settings(storage_backend="chroma", metadata_backend="sqlite")
    app = create_app(settings=settings, storage=FakeStorage())

    payload = {
        "user_id": "user_123",
        "session_id": "sess_xyz",
        "text": "I prefer concise answers.",
        "memory_type_hint": "procedural",
    }

    with TestClient(app) as client:
        response = client.post("/v1/memory", json=payload)

    assert response.status_code == 201
    assert response.json() == {
        "saved": [
            {
                "id": "mem_1",
                "memory_type": "procedural",
                "importance": 0.72,
                "token_count": 34,
                "created_at": "2026-04-19T10:00:00Z",
            }
        ],
        "discarded_count": 0,
    }
    assert FakeMemoryLayer.save_calls[0]["session_id"] == "sess_xyz"


def test_get_memory_recall_returns_recall_result(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _clear_fake_memory_layer()
    monkeypatch.setattr("memory_layer.api.routes.memory.MemoryLayer", FakeMemoryLayer)

    settings = Settings(storage_backend="chroma", metadata_backend="sqlite")
    app = create_app(settings=settings, storage=FakeStorage())

    with TestClient(app) as client:
        response = client.get(
            "/v1/memory/recall",
            params={
                "user_id": "user_123",
                "query": "What stack is the user using?",
                "top_k": 5,
                "token_budget": 200,
                "memory_types": "semantic,procedural",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["total_tokens"] == 24
    assert body["budget_used"] == 0.12
    assert body["prompt_block"] == "<memory>\n[SEMANTIC] User is a backend engineer...\n</memory>"
    assert FakeMemoryLayer.recall_calls[0]["memory_types"] == [
        MemoryType.SEMANTIC,
        MemoryType.PROCEDURAL,
    ]


def test_get_memory_recall_rejects_invalid_memory_types(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _clear_fake_memory_layer()
    monkeypatch.setattr("memory_layer.api.routes.memory.MemoryLayer", FakeMemoryLayer)

    settings = Settings(storage_backend="chroma", metadata_backend="sqlite")
    app = create_app(settings=settings, storage=FakeStorage())

    with TestClient(app) as client:
        response = client.get(
            "/v1/memory/recall",
            params={
                "user_id": "user_123",
                "query": "hello",
                "memory_types": "semantic,unknown",
            },
        )

    assert response.status_code == 422
    assert "Invalid memory_types value(s): unknown" in response.json()["detail"]


def test_post_memory_rate_limit_enforced_per_user(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _clear_fake_memory_layer()
    monkeypatch.setattr("memory_layer.api.routes.memory.MemoryLayer", FakeMemoryLayer)

    settings = Settings(
        storage_backend="chroma",
        metadata_backend="sqlite",
        rate_limit_save=1,
    )
    app = create_app(settings=settings, storage=FakeStorage())

    payload = {
        "user_id": "user_123",
        "session_id": "sess_xyz",
        "text": "I prefer concise answers.",
    }

    with TestClient(app) as client:
        first = client.post("/v1/memory", json=payload)
        second = client.post("/v1/memory", json=payload)

    assert first.status_code == 201
    assert second.status_code == 429
    assert second.json() == {"detail": "Rate limit exceeded"}
    assert second.headers.get("retry-after") is not None
    assert len(FakeMemoryLayer.save_calls) == 1


def test_post_memory_rate_limit_is_user_scoped(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _clear_fake_memory_layer()
    monkeypatch.setattr("memory_layer.api.routes.memory.MemoryLayer", FakeMemoryLayer)

    settings = Settings(
        storage_backend="chroma",
        metadata_backend="sqlite",
        rate_limit_save=1,
    )
    app = create_app(settings=settings, storage=FakeStorage())

    user_a = {
        "user_id": "user_a",
        "session_id": "sess_a",
        "text": "First user payload.",
    }
    user_b = {
        "user_id": "user_b",
        "session_id": "sess_b",
        "text": "Second user payload.",
    }

    with TestClient(app) as client:
        first_a = client.post("/v1/memory", json=user_a)
        first_b = client.post("/v1/memory", json=user_b)

    assert first_a.status_code == 201
    assert first_b.status_code == 201
    assert len(FakeMemoryLayer.save_calls) == 2


def test_get_memory_recall_rate_limit_enforced_per_user(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _clear_fake_memory_layer()
    monkeypatch.setattr("memory_layer.api.routes.memory.MemoryLayer", FakeMemoryLayer)

    settings = Settings(
        storage_backend="chroma",
        metadata_backend="sqlite",
        rate_limit_recall=1,
    )
    app = create_app(settings=settings, storage=FakeStorage())

    params = {
        "user_id": "user_123",
        "query": "What stack is the user using?",
    }

    with TestClient(app) as client:
        first = client.get("/v1/memory/recall", params=params)
        second = client.get("/v1/memory/recall", params=params)

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json() == {"detail": "Rate limit exceeded"}
    assert second.headers.get("retry-after") is not None
    assert len(FakeMemoryLayer.recall_calls) == 1


def test_get_memory_list_returns_paginated_result(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _clear_fake_memory_layer()
    monkeypatch.setattr("memory_layer.api.routes.memory.MemoryLayer", FakeMemoryLayer)

    settings = Settings(storage_backend="chroma", metadata_backend="sqlite")
    app = create_app(settings=settings, storage=FakeStorage())

    create_payloads = [
        {
            "user_id": "user_123",
            "session_id": "sess_1",
            "text": "first entry",
            "memory_type_hint": "semantic",
        },
        {
            "user_id": "user_123",
            "session_id": "sess_2",
            "text": "second entry",
            "memory_type_hint": "procedural",
        },
    ]

    with TestClient(app) as client:
        for payload in create_payloads:
            response = client.post("/v1/memory", json=payload)
            assert response.status_code == 201

        list_response = client.get(
            "/v1/memory",
            params={
                "user_id": "user_123",
                "page": 1,
                "page_size": 20,
                "include_compressed": True,
            },
        )

    assert list_response.status_code == 200
    body = list_response.json()
    assert body["total"] == 2
    assert body["page"] == 1
    assert body["page_size"] == 20
    assert len(body["items"]) == 2
    assert FakeMemoryLayer.list_calls[0]["include_compressed"] is True


def test_delete_memory_and_delete_all_memory(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _clear_fake_memory_layer()
    monkeypatch.setattr("memory_layer.api.routes.memory.MemoryLayer", FakeMemoryLayer)

    settings = Settings(storage_backend="chroma", metadata_backend="sqlite")
    app = create_app(settings=settings, storage=FakeStorage())

    with TestClient(app) as client:
        create_one = client.post(
            "/v1/memory",
            json={
                "user_id": "user_123",
                "session_id": "sess_1",
                "text": "to be removed",
            },
        )
        create_two = client.post(
            "/v1/memory",
            json={
                "user_id": "user_123",
                "session_id": "sess_2",
                "text": "remove with delete all",
            },
        )

        assert create_one.status_code == 201
        assert create_two.status_code == 201

        memory_id = create_one.json()["saved"][0]["id"]
        delete_one = client.delete(
            f"/v1/memory/{memory_id}",
            params={"user_id": "user_123"},
        )
        delete_all = client.request(
            "DELETE",
            "/v1/memory",
            json={
                "user_id": "user_123",
                "confirm": True,
            },
        )

    assert delete_one.status_code == 200
    assert delete_one.json() == {"deleted": True, "id": memory_id}
    assert FakeMemoryLayer.forget_calls[0]["memory_id"] == memory_id

    assert delete_all.status_code == 200
    assert delete_all.json() == {"deleted_count": 1}
    assert FakeMemoryLayer.forget_all_calls[0]["confirm"] is True


def test_delete_all_memory_requires_confirm_true(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _clear_fake_memory_layer()
    monkeypatch.setattr("memory_layer.api.routes.memory.MemoryLayer", FakeMemoryLayer)

    settings = Settings(storage_backend="chroma", metadata_backend="sqlite")
    app = create_app(settings=settings, storage=FakeStorage())

    with TestClient(app) as client:
        response = client.request(
            "DELETE",
            "/v1/memory",
            json={
                "user_id": "user_123",
                "confirm": False,
            },
        )

    assert response.status_code == 422
