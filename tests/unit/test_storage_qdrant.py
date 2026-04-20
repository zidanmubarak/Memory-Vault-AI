from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from memory_vault.models import MemoryChunk, MemoryType
from memory_vault.storage.base import MemorySearchQuery
from memory_vault.storage.qdrant import QdrantAdapter


@dataclass
class FakeSearchHit:
    id: str
    payload: dict[str, Any]
    vector: list[float]
    score: float


@dataclass
class FakePoint:
    id: str
    vector: list[float]
    payload: dict[str, Any]


@dataclass
class FakeQdrantClient:
    collection_exists_flag: bool = False
    vector_size: int | None = None
    points: dict[str, FakePoint] = field(default_factory=dict)

    async def collection_exists(self, *, collection_name: str) -> bool:
        del collection_name
        return self.collection_exists_flag

    async def create_collection(
        self,
        *,
        collection_name: str,
        vectors_config: Any | None = None,
        vector_size: int | None = None,
    ) -> None:
        del collection_name
        self.collection_exists_flag = True
        if vectors_config is not None and hasattr(vectors_config, "size"):
            self.vector_size = int(vectors_config.size)
        elif vector_size is not None:
            self.vector_size = int(vector_size)

    async def upsert(self, *, collection_name: str, points: list[Any], wait: bool = True) -> None:
        del collection_name, wait
        for point in points:
            if isinstance(point, dict):
                point_id = str(point.get("id", ""))
                vector = [float(value) for value in point.get("vector", [])]
                payload_raw = point.get("payload", {})
                payload = payload_raw if isinstance(payload_raw, dict) else {}
            else:
                point_id = str(getattr(point, "id", ""))
                vector = [float(value) for value in getattr(point, "vector", [])]
                payload_raw = getattr(point, "payload", {})
                payload = payload_raw if isinstance(payload_raw, dict) else {}

            if point_id:
                self.points[point_id] = FakePoint(id=point_id, vector=vector, payload=payload)

    async def search(
        self,
        *,
        collection_name: str,
        query_vector: list[float],
        query_filter: Any,
        limit: int,
        with_payload: bool,
        with_vectors: bool,
    ) -> list[FakeSearchHit]:
        del collection_name, with_payload, with_vectors
        user_id = ""
        if isinstance(query_filter, dict):
            user_id = str(query_filter.get("user_id", ""))

        scored: list[FakeSearchHit] = []
        for point in self.points.values():
            if str(point.payload.get("user_id", "")) != user_id:
                continue
            distance = abs(point.vector[0] - query_vector[0])
            score = max(0.0, min(1.0, 1.0 - distance))
            scored.append(
                FakeSearchHit(
                    id=point.id,
                    payload=point.payload,
                    vector=point.vector,
                    score=score,
                )
            )

        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:limit]

    async def scroll(
        self,
        *,
        collection_name: str,
        scroll_filter: Any,
        limit: int,
        offset: Any,
        with_payload: bool,
        with_vectors: bool,
    ) -> tuple[list[FakePoint], Any | None]:
        del collection_name, limit, offset, with_payload, with_vectors
        user_id = ""
        if isinstance(scroll_filter, dict):
            user_id = str(scroll_filter.get("user_id", ""))

        points = [
            point
            for point in self.points.values()
            if str(point.payload.get("user_id", "")) == user_id
        ]
        return points, None

    async def delete(
        self,
        *,
        collection_name: str,
        points_selector: Any | None = None,
        points: list[str] | None = None,
        wait: bool = True,
    ) -> None:
        del collection_name, wait
        ids: list[str] = []

        if points is not None:
            ids = [str(value) for value in points]
        elif points_selector is not None and hasattr(points_selector, "points"):
            ids = [str(value) for value in points_selector.points]

        for point_id in ids:
            self.points.pop(point_id, None)

    async def get_collection(self, *, collection_name: str) -> dict[str, Any]:
        del collection_name
        return {
            "points_count": len(self.points),
            "config": {
                "params": {
                    "vectors": {"size": self.vector_size or 0},
                }
            },
        }


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
async def test_qdrant_adapter_upsert_and_query_filters() -> None:
    client = FakeQdrantClient()
    adapter = QdrantAdapter(
        qdrant_url="http://localhost:6333",
        collection_name="memory_vault",
        client=client,
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
            _chunk(
                memory_id="mem_proc",
                user_id="user_a",
                memory_type=MemoryType.PROCEDURAL,
                embedding=[0.2],
            ),
        ]
    )

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
async def test_qdrant_adapter_delete_methods_are_user_scoped() -> None:
    client = FakeQdrantClient()
    adapter = QdrantAdapter(
        qdrant_url="http://localhost:6333",
        collection_name="memory_vault",
        client=client,
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
