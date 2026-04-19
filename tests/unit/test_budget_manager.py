from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memory_layer.budget.manager import ContextBudgetManager
from memory_layer.models import MemoryChunk, MemoryType


def _chunk(*, memory_id: str, content: str, memory_type: MemoryType) -> MemoryChunk:
    now = datetime(2026, 4, 19, 10, 0, tzinfo=UTC)
    return MemoryChunk(
        id=memory_id,
        user_id="user_a",
        session_id="sess_a",
        memory_type=memory_type,
        content=content,
        importance=0.8,
        token_count=0,
        embedding=None,
        created_at=now,
        updated_at=now,
    )


def test_count_tokens_enforces_minimum_of_one() -> None:
    manager = ContextBudgetManager(token_counter=lambda text: 0)

    assert manager.count_tokens("") == 1


def test_select_validates_inputs() -> None:
    manager = ContextBudgetManager(token_counter=lambda text: len(text.split()))

    with pytest.raises(ValueError):
        manager.select(
            procedural_memories=[],
            ranked_memories=[],
            top_k=0,
            token_budget=10,
        )
    with pytest.raises(ValueError):
        manager.select(
            procedural_memories=[],
            ranked_memories=[],
            top_k=1,
            token_budget=0,
        )


def test_select_prioritizes_procedural_before_ranked() -> None:
    manager = ContextBudgetManager(token_counter=lambda text: len(text.split()))
    procedural = _chunk(
        memory_id="proc",
        content="be concise",
        memory_type=MemoryType.PROCEDURAL,
    )
    ranked = _chunk(
        memory_id="sem",
        content="semantic note",
        memory_type=MemoryType.SEMANTIC,
    )

    selected, tokens = manager.select(
        procedural_memories=[procedural],
        ranked_memories=[ranked],
        top_k=2,
        token_budget=10,
    )

    assert [chunk.id for chunk in selected] == ["proc", "sem"]
    assert tokens == 4


def test_select_skips_oversized_and_keeps_scanning() -> None:
    manager = ContextBudgetManager(token_counter=lambda text: len(text.split()))
    long_chunk = _chunk(
        memory_id="long",
        content="this is too long",
        memory_type=MemoryType.SEMANTIC,
    )
    short_chunk = _chunk(
        memory_id="short",
        content="ok",
        memory_type=MemoryType.SEMANTIC,
    )

    selected, tokens = manager.select(
        procedural_memories=[],
        ranked_memories=[long_chunk, short_chunk],
        top_k=2,
        token_budget=1,
    )

    assert [chunk.id for chunk in selected] == ["short"]
    assert tokens == 1


def test_select_updates_token_count_on_returned_copies() -> None:
    manager = ContextBudgetManager(token_counter=lambda text: len(text.split()))
    original = _chunk(
        memory_id="sem",
        content="one two three",
        memory_type=MemoryType.SEMANTIC,
    )

    selected, _ = manager.select(
        procedural_memories=[],
        ranked_memories=[original],
        top_k=1,
        token_budget=10,
    )

    assert original.token_count == 0
    assert selected[0].token_count == 3


def test_minimum_tokens_uses_smallest_candidate() -> None:
    manager = ContextBudgetManager(token_counter=lambda text: len(text.split()))
    first = _chunk(memory_id="a", content="one two", memory_type=MemoryType.SEMANTIC)
    second = _chunk(memory_id="b", content="one", memory_type=MemoryType.SEMANTIC)

    assert manager.minimum_tokens([first, second]) == 1
    assert manager.minimum_tokens([]) == 1
