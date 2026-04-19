from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from memory_layer.models import MemoryChunk, MemoryType

SAMPLE_MEMORY_PAYLOAD: dict[str, Any] = {
    "id": "mem_a3f9b2c10d4e",
    "user_id": "user_alice",
    "session_id": "sess_9a3b1c2d",
    "memory_type": MemoryType.SEMANTIC,
    "content": "User prefers concise technical responses with examples.",
    "importance": 0.86,
    "token_count": 9,
    "created_at": datetime(2026, 4, 19, 10, 0, tzinfo=UTC),
    "updated_at": datetime(2026, 4, 19, 10, 0, tzinfo=UTC),
}


def sample_memory_chunk() -> MemoryChunk:
    """Return a reusable sample memory chunk for tests."""
    return MemoryChunk(**SAMPLE_MEMORY_PAYLOAD)
