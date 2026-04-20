from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memory_vault.models import MemoryChunk, MemoryType
from memory_vault.prompt.builder import PromptBuilder


def _chunk(*, memory_id: str, memory_type: MemoryType, content: str) -> MemoryChunk:
    now = datetime(2026, 4, 19, 10, 0, tzinfo=UTC)
    return MemoryChunk(
        id=memory_id,
        user_id="user_a",
        session_id="sess_a",
        memory_type=memory_type,
        content=content,
        importance=0.8,
        token_count=2,
        embedding=None,
        created_at=now,
        updated_at=now,
    )


def test_builder_validates_tags() -> None:
    with pytest.raises(ValueError):
        PromptBuilder(start_tag="", end_tag="</memory>")
    with pytest.raises(ValueError):
        PromptBuilder(start_tag="<memory>", end_tag="")


def test_build_empty_memory_block() -> None:
    builder = PromptBuilder()

    block = builder.build([])

    assert block == "<memory>\n</memory>"


def test_build_formats_ordered_memory_lines() -> None:
    builder = PromptBuilder()
    memories = [
        _chunk(
            memory_id="sem",
            memory_type=MemoryType.SEMANTIC,
            content="User builds APIs.",
        ),
        _chunk(
            memory_id="proc",
            memory_type=MemoryType.PROCEDURAL,
            content="Communication style: direct.",
        ),
    ]

    block = builder.build(memories)

    assert block == (
        "<memory>\n"
        "[SEMANTIC] User builds APIs.\n"
        "[PROCEDURAL] Communication style: direct.\n"
        "</memory>"
    )
