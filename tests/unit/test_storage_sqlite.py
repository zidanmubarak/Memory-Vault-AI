from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from memory_layer.models import MemoryChunk, MemoryType
from memory_layer.storage.base import MemoryListQuery, ProceduralMemoryRecord, SessionStatsRecord
from memory_layer.storage.sqlite import SQLiteAdapter


def _memory_chunk(*, memory_id: str, user_id: str, memory_type: MemoryType) -> MemoryChunk:
    return MemoryChunk(
        id=memory_id,
        user_id=user_id,
        session_id="sess_9a3b1c2d",
        memory_type=memory_type,
        content=f"chunk-{memory_id}",
        importance=0.7,
        token_count=4,
        metadata={"source": "test"},
    )


@pytest.mark.asyncio
async def test_sqlite_adapter_chunk_crud(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    adapter = SQLiteAdapter(sqlite_path=str(db_path))

    await adapter.initialize()

    chunks = [
        _memory_chunk(memory_id="mem_a", user_id="user_a", memory_type=MemoryType.SEMANTIC),
        _memory_chunk(memory_id="mem_b", user_id="user_a", memory_type=MemoryType.EPISODIC),
    ]
    await adapter.upsert_memory_chunks(chunks)

    fetched = await adapter.get_memory_chunk(memory_id="mem_a", user_id="user_a")
    assert fetched is not None
    assert fetched.id == "mem_a"

    listed = await adapter.list_memory_chunks(
        MemoryListQuery(user_id="user_a", page=1, page_size=10)
    )
    assert listed.total == 2
    assert len(listed.items) == 2

    deleted_one = await adapter.delete_memory_chunk(memory_id="mem_a", user_id="user_a")
    assert deleted_one is True

    deleted_all = await adapter.delete_memory_chunks_for_user(user_id="user_a")
    assert deleted_all == 1

    await adapter.close()


@pytest.mark.asyncio
async def test_sqlite_adapter_procedural_and_session_tables(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "memory.db"
    adapter = SQLiteAdapter(sqlite_path=str(db_path))

    await adapter.initialize()

    record = ProceduralMemoryRecord(
        user_id="user_a",
        key="communication_style",
        value="concise",
        confidence=0.9,
    )
    await adapter.upsert_procedural_memory(record)

    procedural = await adapter.list_procedural_memory(user_id="user_a")
    assert len(procedural) == 1
    assert procedural[0].value == "concise"

    removed = await adapter.delete_procedural_memory(user_id="user_a", key="communication_style")
    assert removed is True

    stats = SessionStatsRecord(
        session_id="sess_9a3b1c2d",
        user_id="user_a",
        memory_count=2,
        total_tokens_stored=10,
        started_at=datetime(2026, 4, 19, 10, 0, tzinfo=UTC),
        last_activity=datetime(2026, 4, 19, 10, 5, tzinfo=UTC),
    )
    await adapter.upsert_session_stats(stats)

    fetched = await adapter.get_session_stats(session_id="sess_9a3b1c2d")
    assert fetched is not None
    assert fetched.user_id == "user_a"
    assert fetched.total_tokens_stored == 10

    health = await adapter.healthcheck()
    assert health["status"] == "ok"

    await adapter.close()
