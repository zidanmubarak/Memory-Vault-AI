from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from memory_layer.models import MemoryChunk, MemoryType
from memory_layer.storage.base import MemorySearchQuery
from memory_layer.storage.chroma import ChromaAdapter


@dataclass
class _StoredVector:
    embedding: list[float]
    document: str
    metadata: dict[str, Any]


@dataclass
class FakeChromaCollection:
    items: dict[str, _StoredVector] = field(default_factory=dict)

    def upsert(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float] | None],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        for index, memory_id in enumerate(ids):
            embedding = embeddings[index]
            if embedding is None:
                continue
            self.items[memory_id] = _StoredVector(
                embedding=list(embedding),
                document=documents[index],
                metadata=metadatas[index],
            )

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict[str, Any],
        include: list[str],
    ) -> dict[str, Any]:
        del include
        target = query_embeddings[0]
        user_id = str(where["user_id"])

        scored: list[tuple[str, _StoredVector, float]] = []
        for memory_id, stored in self.items.items():
            if str(stored.metadata.get("user_id", "")) != user_id:
                continue
            distance = abs(stored.embedding[0] - target[0])
            scored.append((memory_id, stored, distance))

        scored.sort(key=lambda row: row[2])
        selected = scored[:n_results]

        return {
            "ids": [[row[0] for row in selected]],
            "documents": [[row[1].document for row in selected]],
            "embeddings": [[row[1].embedding for row in selected]],
            "metadatas": [[row[1].metadata for row in selected]],
            "distances": [[row[2] for row in selected]],
        }

    def get(
        self,
        *,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        del include
        selected: list[tuple[str, _StoredVector]] = []

        if ids is not None:
            for memory_id in ids:
                if memory_id in self.items:
                    selected.append((memory_id, self.items[memory_id]))
        elif where is not None:
            user_id = str(where.get("user_id", ""))
            for memory_id, stored in self.items.items():
                if str(stored.metadata.get("user_id", "")) == user_id:
                    selected.append((memory_id, stored))

        return {
            "ids": [memory_id for memory_id, _ in selected],
            "metadatas": [stored.metadata for _, stored in selected],
        }

    def delete(
        self,
        *,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> None:
        if ids is not None:
            for memory_id in ids:
                self.items.pop(memory_id, None)
            return

        if where is not None:
            user_id = str(where.get("user_id", ""))
            to_delete = [
                memory_id
                for memory_id, stored in self.items.items()
                if str(stored.metadata.get("user_id", "")) == user_id
            ]
            for memory_id in to_delete:
                self.items.pop(memory_id, None)

    def count(self) -> int:
        return len(self.items)


@dataclass
class FakeChromaClient:
    collection: FakeChromaCollection

    def get_or_create_collection(self, *, name: str) -> FakeChromaCollection:
        del name
        return self.collection


def _chunk(
    *,
    memory_id: str,
    user_id: str,
    memory_type: MemoryType,
    embedding: list[float],
    compressed: bool = False,
) -> MemoryChunk:
    return MemoryChunk(
        id=memory_id,
        user_id=user_id,
        session_id="sess_9a3b1c2d",
        memory_type=memory_type,
        content=f"content-{memory_id}",
        importance=0.9,
        token_count=3,
        embedding=embedding,
        compressed=compressed,
    )


@pytest.mark.asyncio
async def test_chroma_adapter_upsert_and_query_filters() -> None:
    collection = FakeChromaCollection()
    adapter = ChromaAdapter(
        chroma_path="./data/chroma",
        client=FakeChromaClient(collection),
    )

    chunks = [
        _chunk(
            memory_id="mem_a",
            user_id="user_a",
            memory_type=MemoryType.SEMANTIC,
            embedding=[0.1],
        ),
        _chunk(
            memory_id="mem_b",
            user_id="user_a",
            memory_type=MemoryType.EPISODIC,
            embedding=[0.9],
            compressed=True,
        ),
        _chunk(
            memory_id="mem_c",
            user_id="user_b",
            memory_type=MemoryType.SEMANTIC,
            embedding=[0.1],
        ),
    ]

    await adapter.initialize()
    await adapter.upsert_vectors(chunks)

    result = await adapter.query_vectors(
        MemorySearchQuery(
            user_id="user_a",
            query_embedding=[0.1],
            top_k=5,
            memory_types=(MemoryType.SEMANTIC,),
            include_compressed=False,
            min_importance=0.2,
        )
    )

    assert [chunk.id for chunk in result] == ["mem_a"]
    assert result[0].relevance_score is not None
    assert result[0].embedding == [0.1]


@pytest.mark.asyncio
async def test_chroma_adapter_delete_methods_are_user_scoped() -> None:
    collection = FakeChromaCollection()
    adapter = ChromaAdapter(
        chroma_path="./data/chroma",
        client=FakeChromaClient(collection),
    )

    await adapter.initialize()
    await adapter.upsert_vectors(
        [
            _chunk(
                memory_id="mem_a",
                user_id="user_a",
                memory_type=MemoryType.SEMANTIC,
                embedding=[0.1],
            ),
            _chunk(
                memory_id="mem_b",
                user_id="user_a",
                memory_type=MemoryType.SEMANTIC,
                embedding=[0.2],
            ),
            _chunk(
                memory_id="mem_c",
                user_id="user_b",
                memory_type=MemoryType.SEMANTIC,
                embedding=[0.3],
            ),
        ]
    )

    deleted = await adapter.delete_vectors(["mem_a", "mem_c"], user_id="user_a")
    assert deleted == 1

    deleted_for_user = await adapter.delete_vectors_for_user(user_id="user_a")
    assert deleted_for_user == 1

    health = await adapter.healthcheck()
    assert health["status"] == "ok"
    assert health["documents"] == "1"
