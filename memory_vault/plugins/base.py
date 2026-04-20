from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

from memory_vault.models import MemoryType


class MemoryTypePlugin(ABC):
    """Base class for custom memory type plugins used during ingestion."""

    name: str
    base_memory_type: MemoryType
    priority: int = 100

    @abstractmethod
    def matches(self, chunk_text: str) -> bool:
        """Return True when this plugin should classify the chunk."""

    def metadata(self, chunk_text: str) -> Mapping[str, object] | None:
        """Return optional metadata to attach to chunks classified by this plugin."""
        del chunk_text
        return None


__all__ = ["MemoryTypePlugin"]
