from __future__ import annotations

from memory_vault.storage import ChromaAdapter, CompositeStorage, QdrantAdapter, SQLiteAdapter


def test_storage_init_exports_concrete_adapters() -> None:
    assert ChromaAdapter.__name__ == "ChromaAdapter"
    assert QdrantAdapter.__name__ == "QdrantAdapter"
    assert SQLiteAdapter.__name__ == "SQLiteAdapter"
    assert CompositeStorage.__name__ == "CompositeStorage"
