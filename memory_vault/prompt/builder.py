from __future__ import annotations

from collections.abc import Sequence

from memory_vault.models import MemoryChunk


class PromptBuilder:
    """Build XML-wrapped memory prompt blocks for downstream LLM calls."""

    def __init__(self, *, start_tag: str = "<memory>", end_tag: str = "</memory>") -> None:
        if not start_tag:
            raise ValueError("start_tag is required")
        if not end_tag:
            raise ValueError("end_tag is required")

        self._start_tag = start_tag
        self._end_tag = end_tag

    def build(self, memories: Sequence[MemoryChunk]) -> str:
        """Build the memory prompt block preserving memory order."""
        lines = [self._start_tag]
        for memory in memories:
            lines.append(f"[{self._memory_label(memory)}] {memory.content}")
        lines.append(self._end_tag)
        return "\n".join(lines)

    @staticmethod
    def _memory_label(memory: MemoryChunk) -> str:
        """Build prompt label including optional custom memory type marker."""
        base_label = memory.memory_type.value.upper()
        custom_type = memory.metadata.get("custom_memory_type")
        if not isinstance(custom_type, str):
            return base_label

        normalized = custom_type.strip().replace(" ", "_")
        if not normalized:
            return base_label
        return f"{base_label}:{normalized.upper()}"


__all__ = ["PromptBuilder"]
