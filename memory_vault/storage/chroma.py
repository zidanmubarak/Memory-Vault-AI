from __future__ import annotations

import importlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from functools import partial
from typing import Any

import anyio

from memory_vault.exceptions import ConfigurationError, StorageError
from memory_vault.models import MemoryChunk, MemoryType
from memory_vault.storage.base import MemorySearchQuery, VectorStoreBackend


def _to_utc_datetime(value: Any) -> datetime:
    """Convert an arbitrary timestamp value into a UTC datetime."""
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).astimezone(UTC)
    return datetime.now(UTC)


def _to_float(value: Any, *, default: float = 0.0) -> float:
    """Safely cast an arbitrary value to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, *, default: int = 0) -> int:
    """Safely cast an arbitrary value to int."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any, *, default: bool = False) -> bool:
    """Safely cast an arbitrary value to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes"}:
            return True
        if lowered in {"0", "false", "no"}:
            return False
    return default


def _distance_to_score(distance: Any) -> float | None:
    """Map a cosine distance to a bounded similarity score."""
    if distance is None:
        return None
    score = 1.0 - _to_float(distance, default=1.0)
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _to_embedding(value: Any) -> list[float] | None:
    """Convert a raw embedding payload to a float list when available."""
    if value is None:
        return None

    try:
        return [float(component) for component in value]
    except (TypeError, ValueError):
        return None


class ChromaAdapter(VectorStoreBackend):
    """Chroma-backed vector store adapter."""

    def __init__(
        self,
        *,
        chroma_path: str,
        collection_name: str = "memory_vault",
        client: Any | None = None,
        collection: Any | None = None,
    ) -> None:
        self._chroma_path = chroma_path
        self._collection_name = collection_name
        self._client = client
        self._collection = collection
        self._owns_client = client is None
        self._initialized = collection is not None

    async def initialize(self) -> None:
        """Initialize and open the configured Chroma collection."""
        if self._initialized:
            return

        if self._collection is not None:
            self._initialized = True
            return

        if self._client is None:
            try:
                chromadb_module = importlib.import_module("chromadb")
            except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional deps
                raise ConfigurationError(
                    "ChromaAdapter requires chromadb. Install memory-vault-ai[storage]."
                ) from exc
            create_client = partial(chromadb_module.PersistentClient, path=self._chroma_path)
            self._client = await anyio.to_thread.run_sync(create_client)

        client = self._client
        if client is None:
            raise StorageError("Chroma client is not initialized")

        try:
            get_collection = partial(
                client.get_or_create_collection,
                name=self._collection_name,
            )
            self._collection = await anyio.to_thread.run_sync(get_collection)
        except Exception as exc:  # pragma: no cover - defensive error wrapping
            raise StorageError(f"Failed to initialize Chroma collection: {exc}") from exc

        self._initialized = True

    async def close(self) -> None:
        """Close adapter resources."""
        self._collection = None
        self._initialized = False
        if self._owns_client:
            self._client = None

    async def healthcheck(self) -> dict[str, str]:
        """Return health information for the Chroma backend."""
        await self.initialize()
        if self._collection is None:
            raise StorageError("Chroma collection is not initialized")

        try:
            count_operation = partial(self._collection.count)
            count = await anyio.to_thread.run_sync(count_operation)
        except Exception as exc:  # pragma: no cover - defensive error wrapping
            raise StorageError(f"Chroma healthcheck failed: {exc}") from exc

        return {
            "status": "ok",
            "backend": "chroma",
            "collection": self._collection_name,
            "documents": str(count),
        }

    async def upsert_vectors(self, chunks: Sequence[MemoryChunk]) -> None:
        """Persist vector entries for chunks with embeddings."""
        await self.initialize()
        if self._collection is None:
            raise StorageError("Chroma collection is not initialized")

        vector_chunks = [
            chunk
            for chunk in chunks
            if chunk.embedding is not None and chunk.memory_type is not MemoryType.PROCEDURAL
        ]
        if not vector_chunks:
            return

        ids = [chunk.id for chunk in vector_chunks]
        embeddings = [chunk.embedding for chunk in vector_chunks]
        documents = [chunk.content for chunk in vector_chunks]
        metadatas = [
            {
                "user_id": chunk.user_id,
                "session_id": chunk.session_id,
                "memory_type": chunk.memory_type.value,
                "importance": chunk.importance,
                "token_count": chunk.token_count,
                "compressed": chunk.compressed,
                "compression_source": chunk.compression_source,
                "source_session_id": chunk.source_session_id or "",
                "created_at": chunk.created_at.astimezone(UTC).isoformat(),
                "updated_at": chunk.updated_at.astimezone(UTC).isoformat(),
                "metadata_json": json.dumps(chunk.metadata, separators=(",", ":")),
            }
            for chunk in vector_chunks
        ]

        try:
            upsert_operation = partial(
                self._collection.upsert,
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            await anyio.to_thread.run_sync(upsert_operation)
        except Exception as exc:  # pragma: no cover - defensive error wrapping
            raise StorageError(f"Failed to upsert vectors into Chroma: {exc}") from exc

    async def query_vectors(self, query: MemorySearchQuery) -> list[MemoryChunk]:
        """Run a user-scoped vector query and return mapped memory chunks."""
        await self.initialize()
        if self._collection is None:
            raise StorageError("Chroma collection is not initialized")

        try:
            query_operation = partial(
                self._collection.query,
                query_embeddings=[list(query.query_embedding)],
                n_results=query.top_k,
                where={"user_id": query.user_id},
                include=["documents", "embeddings", "metadatas", "distances"],
            )
            raw = await anyio.to_thread.run_sync(query_operation)
        except Exception as exc:  # pragma: no cover - defensive error wrapping
            raise StorageError(f"Failed to query vectors from Chroma: {exc}") from exc

        ids_nested = raw.get("ids") or [[]]
        docs_nested = raw.get("documents") or [[]]
        embeddings_nested = raw.get("embeddings") or [[]]
        metas_nested = raw.get("metadatas") or [[]]
        distances_nested = raw.get("distances") or [[]]

        ids = ids_nested[0] if ids_nested else []
        documents = docs_nested[0] if docs_nested else []
        embeddings = embeddings_nested[0] if embeddings_nested else []
        metadatas = metas_nested[0] if metas_nested else []
        distances = distances_nested[0] if distances_nested else []

        candidates: list[MemoryChunk] = []
        for index, memory_id in enumerate(ids):
            metadata: dict[str, Any] = metadatas[index] if index < len(metadatas) else {}
            content = documents[index] if index < len(documents) else ""
            if not content:
                continue

            memory_type_raw = str(metadata.get("memory_type", MemoryType.EPISODIC.value))
            try:
                memory_type = MemoryType(memory_type_raw)
            except ValueError:
                memory_type = MemoryType.EPISODIC

            raw_metadata = metadata.get("metadata_json", "{}")
            parsed_metadata: dict[str, Any]
            if isinstance(raw_metadata, str):
                try:
                    parsed_metadata = json.loads(raw_metadata)
                except json.JSONDecodeError:
                    parsed_metadata = {}
            elif isinstance(raw_metadata, dict):
                parsed_metadata = raw_metadata
            else:
                parsed_metadata = {}

            chunk = MemoryChunk(
                id=str(memory_id),
                user_id=str(metadata.get("user_id", query.user_id)),
                session_id=str(metadata.get("session_id", "unknown_session")),
                memory_type=memory_type,
                content=content,
                importance=_to_float(metadata.get("importance"), default=0.0),
                token_count=_to_int(metadata.get("token_count"), default=0),
                embedding=(
                    _to_embedding(embeddings[index]) if index < len(embeddings) else None
                ),
                compressed=_to_bool(metadata.get("compressed"), default=False),
                compression_source=_to_bool(metadata.get("compression_source"), default=False),
                source_session_id=(
                    str(metadata.get("source_session_id"))
                    if metadata.get("source_session_id")
                    else None
                ),
                chroma_id=str(memory_id),
                created_at=_to_utc_datetime(metadata.get("created_at")),
                updated_at=_to_utc_datetime(metadata.get("updated_at")),
                metadata=parsed_metadata,
                relevance_score=(
                    _distance_to_score(distances[index]) if index < len(distances) else None
                ),
            )

            if not query.include_compressed and chunk.compressed:
                continue
            if chunk.importance < query.min_importance:
                continue
            if query.memory_types and chunk.memory_type not in query.memory_types:
                continue
            candidates.append(chunk)

        candidates.sort(key=lambda chunk: chunk.relevance_score or 0.0, reverse=True)
        return candidates[: query.top_k]

    async def delete_vectors(self, memory_ids: Sequence[str], *, user_id: str) -> int:
        """Delete selected vectors owned by the specified user."""
        await self.initialize()
        if self._collection is None:
            raise StorageError("Chroma collection is not initialized")
        if not memory_ids:
            return 0

        try:
            get_operation = partial(
                self._collection.get,
                ids=list(memory_ids),
                include=["metadatas"],
            )
            raw = await anyio.to_thread.run_sync(get_operation)
        except Exception as exc:  # pragma: no cover - defensive error wrapping
            raise StorageError(f"Failed to read vectors before delete: {exc}") from exc

        raw_ids = raw.get("ids") or []
        raw_metadatas = raw.get("metadatas") or []

        allowed_ids: list[str] = []
        for index, candidate_id in enumerate(raw_ids):
            metadata = raw_metadatas[index] if index < len(raw_metadatas) else {}
            if str(metadata.get("user_id", "")) == user_id:
                allowed_ids.append(str(candidate_id))

        if not allowed_ids:
            return 0

        try:
            delete_operation = partial(self._collection.delete, ids=allowed_ids)
            await anyio.to_thread.run_sync(delete_operation)
        except Exception as exc:  # pragma: no cover - defensive error wrapping
            raise StorageError(f"Failed to delete vectors from Chroma: {exc}") from exc

        return len(allowed_ids)

    async def delete_vectors_for_user(self, *, user_id: str) -> int:
        """Delete all vectors for the given user."""
        await self.initialize()
        if self._collection is None:
            raise StorageError("Chroma collection is not initialized")

        try:
            get_operation = partial(self._collection.get, where={"user_id": user_id}, include=[])
            raw = await anyio.to_thread.run_sync(get_operation)
        except Exception as exc:  # pragma: no cover - defensive error wrapping
            raise StorageError(f"Failed to list vectors for delete: {exc}") from exc

        raw_ids = raw.get("ids") or []
        if raw_ids and isinstance(raw_ids[0], list):
            ids_to_delete = [str(item) for nested in raw_ids for item in nested]
        else:
            ids_to_delete = [str(item) for item in raw_ids]

        if not ids_to_delete:
            return 0

        try:
            delete_operation = partial(self._collection.delete, where={"user_id": user_id})
            await anyio.to_thread.run_sync(delete_operation)
        except Exception as exc:  # pragma: no cover - defensive error wrapping
            raise StorageError(f"Failed to delete user vectors from Chroma: {exc}") from exc

        return len(ids_to_delete)


__all__ = ["ChromaAdapter"]
