from __future__ import annotations

from datetime import UTC, datetime

from memory_vault.models import MemoryChunk, MemoryType
from memory_vault.prompt.builder import PromptBuilder


def test_build_formats_custom_memory_type_label() -> None:
    now = datetime(2026, 4, 19, 10, 0, tzinfo=UTC)
    memory = MemoryChunk(
        id="mem_custom",
        user_id="user_a",
        session_id="sess_a",
        memory_type=MemoryType.SEMANTIC,
        content="Project profile summary.",
        importance=0.8,
        token_count=3,
        created_at=now,
        updated_at=now,
        metadata={"custom_memory_type": "project_profile"},
    )

    block = PromptBuilder().build([memory])

    assert block == (
        "<memory>\n"
        "[SEMANTIC:PROJECT_PROFILE] Project profile summary.\n"
        "</memory>"
    )
