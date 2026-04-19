from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from memory_layer.models import (
    MemoryChunk,
    MemoryConfig,
    MemoryType,
    PaginatedResult,
    RecallResult,
)


def test_memory_chunk_defaults_are_valid() -> None:
    chunk = MemoryChunk(
        id="mem_a3f9b2c10d4e",
        user_id="user_alice",
        session_id="sess_9a3b1c2d",
        memory_type=MemoryType.SEMANTIC,
        content="User is building a FastAPI application.",
        importance=0.72,
        token_count=8,
    )

    assert chunk.compressed is False
    assert chunk.metadata == {}
    assert chunk.created_at.tzinfo == UTC
    assert chunk.updated_at >= chunk.created_at


def test_memory_chunk_rejects_invalid_importance() -> None:
    with pytest.raises(ValidationError):
        MemoryChunk(
            id="mem_a3f9b2c10d4e",
            user_id="user_alice",
            session_id="sess_9a3b1c2d",
            memory_type=MemoryType.SEMANTIC,
            content="Invalid score.",
            importance=1.2,
            token_count=3,
        )


def test_memory_chunk_requires_source_session_for_compression_source() -> None:
    with pytest.raises(ValidationError):
        MemoryChunk(
            id="mem_a3f9b2c10d4e",
            user_id="user_alice",
            session_id="sess_9a3b1c2d",
            memory_type=MemoryType.SEMANTIC,
            content="Compressed summary.",
            importance=0.8,
            token_count=4,
            compression_source=True,
        )


def test_memory_chunk_rejects_out_of_order_timestamps() -> None:
    with pytest.raises(ValidationError):
        MemoryChunk(
            id="mem_a3f9b2c10d4e",
            user_id="user_alice",
            session_id="sess_9a3b1c2d",
            memory_type=MemoryType.EPISODIC,
            content="Out of order timestamps.",
            importance=0.8,
            token_count=4,
            created_at=datetime(2026, 1, 2, tzinfo=UTC),
            updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        )


def test_memory_config_validates_backend_dependencies() -> None:
    with pytest.raises(ValidationError):
        MemoryConfig(storage_backend="qdrant")


def test_memory_config_validates_chunk_bounds() -> None:
    with pytest.raises(ValidationError):
        MemoryConfig(min_chunk_tokens=400, max_chunk_tokens=100)


def test_paginated_result_computes_total_pages() -> None:
    page = PaginatedResult[MemoryChunk](items=[], total=41, page=1, page_size=20)

    assert page.total_pages == 3


def test_recall_result_budget_bounds() -> None:
    with pytest.raises(ValidationError):
        RecallResult(memories=[], total_tokens=0, budget_used=1.1)
