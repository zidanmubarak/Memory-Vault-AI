from __future__ import annotations

from collections.abc import Sequence

import pytest
from fastapi.testclient import TestClient

from memory_vault.api.main import create_app
from memory_vault.config import Settings
from memory_vault.exceptions import ConfigurationError
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
        self.health_payload = {
            "status": "ok",
            "vector": "ok",
            "metadata": "ok",
        }

    async def initialize(self) -> None:
        self.initialized = True

    async def close(self) -> None:
        self.closed = True

    async def healthcheck(self) -> dict[str, str]:
        return self.health_payload

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


def test_create_app_health_endpoint_and_lifespan() -> None:
    settings = Settings(
        embedding_model="test-embedder",
        storage_backend="chroma",
        metadata_backend="sqlite",
    )
    storage = FakeStorage()
    app = create_app(settings=settings, storage=storage)

    with TestClient(app) as client:
        response = client.get("/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": "0.1.0.dev0",
        "storage": {"chroma": "ok", "sqlite": "ok"},
        "embedding_model": "test-embedder",
    }
    assert storage.initialized is True
    assert storage.closed is True


def test_create_app_supports_qdrant_storage_backend() -> None:
    settings = Settings(
        storage_backend="qdrant",
        qdrant_url="http://localhost:6333",
        metadata_backend="sqlite",
    )

    app = create_app(settings=settings)

    assert app.title == "Memory Vault AI"


def test_create_app_raises_for_unsupported_metadata_backend() -> None:
    settings = Settings(
        storage_backend="chroma",
        metadata_backend="postgres",
        postgres_url="postgresql://localhost/test",
    )

    with pytest.raises(ConfigurationError):
        create_app(settings=settings)


def test_create_app_enforces_api_key_on_v1_routes() -> None:
    settings = Settings(
        storage_backend="chroma",
        metadata_backend="sqlite",
        api_key="top-secret-key",
    )
    app = create_app(settings=settings, storage=FakeStorage())

    with TestClient(app) as client:
        missing_header = client.get("/v1/session/sess_missing/stats?user_id=user_a")
        wrong_scheme = client.get(
            "/v1/session/sess_missing/stats?user_id=user_a",
            headers={"Authorization": "Token top-secret-key"},
        )
        wrong_key = client.get(
            "/v1/session/sess_missing/stats?user_id=user_a",
            headers={"Authorization": "Bearer wrong"},
        )

    assert missing_header.status_code == 401
    assert missing_header.headers.get("www-authenticate") == "Bearer"
    assert missing_header.json() == {"detail": "Unauthorized"}

    assert wrong_scheme.status_code == 401
    assert wrong_scheme.json() == {"detail": "Unauthorized"}

    assert wrong_key.status_code == 401
    assert wrong_key.json() == {"detail": "Unauthorized"}


def test_create_app_allows_authorized_requests_with_api_key() -> None:
    settings = Settings(
        storage_backend="chroma",
        metadata_backend="sqlite",
        api_key="top-secret-key",
    )
    app = create_app(settings=settings, storage=FakeStorage())

    with TestClient(app) as client:
        protected_response = client.get(
            "/v1/session/sess_missing/stats?user_id=user_a",
            headers={"Authorization": "Bearer top-secret-key"},
        )
        health_response = client.get("/v1/health")

    assert protected_response.status_code == 404
    assert protected_response.json() == {"detail": "Session not found"}

    assert health_response.status_code == 200


def test_create_app_publishes_openapi_schema_and_docs() -> None:
    settings = Settings(storage_backend="chroma", metadata_backend="sqlite")
    app = create_app(settings=settings, storage=FakeStorage())

    with TestClient(app) as client:
        docs_response = client.get("/docs")
        schema_response = client.get("/openapi.json")

    assert docs_response.status_code == 200
    assert "Swagger UI" in docs_response.text

    assert schema_response.status_code == 200
    schema = schema_response.json()
    paths = schema.get("paths", {})
    assert "/v1/memory" in paths
    assert "/v1/memory/recall" in paths
    assert "/v1/session/{session_id}/stats" in paths


def test_create_app_disables_metrics_endpoint_by_default() -> None:
    settings = Settings(storage_backend="chroma", metadata_backend="sqlite")
    app = create_app(settings=settings, storage=FakeStorage())

    with TestClient(app) as client:
        metrics_response = client.get("/metrics")

    assert metrics_response.status_code == 404


def test_create_app_exposes_prometheus_metrics_when_enabled() -> None:
    settings = Settings(
        storage_backend="chroma",
        metadata_backend="sqlite",
        metrics_enabled=True,
    )
    app = create_app(settings=settings, storage=FakeStorage())

    with TestClient(app) as client:
        health_response = client.get("/v1/health")
        metrics_response = client.get("/metrics")

    assert health_response.status_code == 200
    assert metrics_response.status_code == 200
    assert metrics_response.headers["content-type"].startswith("text/plain")

    metrics_body = metrics_response.text
    assert "memory_vault_requests_total" in metrics_body
    assert "memory_vault_request_duration_seconds" in metrics_body
    assert "memory_vault_memories_total" in metrics_body
    assert "memory_vault_recall_latency_seconds" in metrics_body
    assert "memory_vault_ingestion_latency_seconds" in metrics_body
    assert "memory_vault_token_budget_utilization" in metrics_body
    assert 'endpoint="/v1/health"' in metrics_body


def test_create_app_serves_memory_introspection_ui() -> None:
    settings = Settings(storage_backend="chroma", metadata_backend="sqlite")
    app = create_app(settings=settings, storage=FakeStorage())

    with TestClient(app) as client:
        response = client.get("/ui")

    assert response.status_code == 200
    assert "Memory Introspection Console" in response.text
    assert "/v1/memory" in response.text
