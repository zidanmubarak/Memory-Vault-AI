from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence

from memory_vault.models import MemoryChunk


def _whitespace_token_count(text: str) -> int:
    """Return a fallback token estimate based on whitespace splitting."""
    return len(text.split())


def _build_tiktoken_counter(encoding_name: str) -> Callable[[str], int] | None:
    """Build a tiktoken-based token counter when dependency is available."""
    try:
        tiktoken = importlib.import_module("tiktoken")
    except ModuleNotFoundError:
        return None

    encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(text: str) -> int:
        return len(encoding.encode(text))

    return count_tokens


class ContextBudgetManager:
    """Enforce per-call token budgets when selecting retrieved memories."""

    def __init__(
        self,
        *,
        token_counter: Callable[[str], int] | None = None,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self._token_count = (
            token_counter
            or _build_tiktoken_counter(encoding_name)
            or _whitespace_token_count
        )

    def select(
        self,
        *,
        procedural_memories: Sequence[MemoryChunk],
        ranked_memories: Sequence[MemoryChunk],
        top_k: int,
        token_budget: int,
    ) -> tuple[list[MemoryChunk], int]:
        """Select memories in priority order while enforcing token and count limits."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if token_budget <= 0:
            raise ValueError("token_budget must be greater than zero")

        selected: list[MemoryChunk] = []
        tokens_used = 0

        for chunk in [*procedural_memories, *ranked_memories]:
            if len(selected) >= top_k:
                break

            chunk_tokens = self.count_tokens(chunk.content)
            if tokens_used + chunk_tokens > token_budget:
                continue

            selected.append(chunk.model_copy(update={"token_count": chunk_tokens}))
            tokens_used += chunk_tokens

        return selected, tokens_used

    def minimum_tokens(self, memories: Sequence[MemoryChunk]) -> int:
        """Return minimum token count among available candidates."""
        if not memories:
            return 1

        return min(self.count_tokens(memory.content) for memory in memories)

    def count_tokens(self, text: str) -> int:
        """Return token count for text with minimum value of one token."""
        count = self._token_count(text)
        return count if count > 0 else 1


__all__ = ["ContextBudgetManager"]
