from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from memory_layer.models import MemoryType
from memory_layer.storage.base import (
    BackendLifecycle,
    MemoryListQuery,
    MemorySearchQuery,
    MetadataStoreBackend,
    ProceduralMemoryRecord,
    SessionStatsRecord,
    StorageBackend,
    VectorStoreBackend,
)


class FakeLifecycleBackend(BackendLifecycle):
    def __init__(self) -> None:
        self.initialized = False
        self.closed = False

    async def initialize(self) -> None:
        self.initialized = True

    async def close(self) -> None:
        self.closed = True

    async def healthcheck(self) -> dict[str, str]:
        return {"status": "ok"}


@pytest.mark.asyncio
async def test_backend_lifecycle_context_manager() -> None:
    backend = FakeLifecycleBackend()

    async with backend:
        assert backend.initialized is True
        assert backend.closed is False

    assert backend.closed is True


def _instantiate_backend(cls: type[Any]) -> object:
    return cls()


def test_memory_search_query_validation() -> None:
    query = MemorySearchQuery(user_id="user_alice", query_embedding=[0.1, 0.2])

    assert query.top_k == 20
    assert query.min_importance == 0.0

    with pytest.raises(ValueError):
        MemorySearchQuery(user_id="", query_embedding=[0.1])
    with pytest.raises(ValueError):
        MemorySearchQuery(user_id="user_alice", query_embedding=[])
    with pytest.raises(ValueError):
        MemorySearchQuery(user_id="user_alice", query_embedding=[0.1], top_k=0)
    with pytest.raises(ValueError):
        MemorySearchQuery(
            user_id="user_alice",
            query_embedding=[0.1],
            min_importance=1.5,
        )


def test_memory_list_query_validation() -> None:
    query = MemoryListQuery(user_id="user_alice", memory_type=MemoryType.SEMANTIC)

    assert query.page == 1
    assert query.page_size == 20

    with pytest.raises(ValueError):
        MemoryListQuery(user_id="", page=1, page_size=10)
    with pytest.raises(ValueError):
        MemoryListQuery(user_id="user_alice", page=0, page_size=10)
    with pytest.raises(ValueError):
        MemoryListQuery(user_id="user_alice", page=1, page_size=0)


def test_procedural_memory_record_validation() -> None:
    record = ProceduralMemoryRecord(
        user_id="user_alice",
        key="communication_style",
        value="concise",
    )

    assert record.confidence == 1.0

    with pytest.raises(ValueError):
        ProceduralMemoryRecord(user_id="", key="k", value="v")
    with pytest.raises(ValueError):
        ProceduralMemoryRecord(user_id="u", key="", value="v")
    with pytest.raises(ValueError):
        ProceduralMemoryRecord(user_id="u", key="k", value="")
    with pytest.raises(ValueError):
        ProceduralMemoryRecord(user_id="u", key="k", value="v", confidence=1.1)


def test_session_stats_record_validation() -> None:
    started_at = datetime(2026, 4, 19, 10, 0, tzinfo=UTC)
    last_activity = datetime(2026, 4, 19, 10, 5, tzinfo=UTC)

    stats = SessionStatsRecord(
        session_id="sess_9a3b1c2d",
        user_id="user_alice",
        memory_count=2,
        total_tokens_stored=120,
        started_at=started_at,
        last_activity=last_activity,
    )

    assert stats.compressed is False

    with pytest.raises(ValueError):
        SessionStatsRecord(session_id="", user_id="user_alice")
    with pytest.raises(ValueError):
        SessionStatsRecord(session_id="sess", user_id="")
    with pytest.raises(ValueError):
        SessionStatsRecord(session_id="sess", user_id="u", memory_count=-1)
    with pytest.raises(ValueError):
        SessionStatsRecord(session_id="sess", user_id="u", total_tokens_stored=-1)
    with pytest.raises(ValueError):
        SessionStatsRecord(
            session_id="sess",
            user_id="u",
            started_at=last_activity,
            last_activity=started_at,
        )


def test_storage_interfaces_are_abstract() -> None:
    with pytest.raises(TypeError):
        _instantiate_backend(VectorStoreBackend)
    with pytest.raises(TypeError):
        _instantiate_backend(MetadataStoreBackend)
    with pytest.raises(TypeError):
        _instantiate_backend(StorageBackend)
