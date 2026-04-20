from __future__ import annotations

import importlib
import inspect
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

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


def _to_score(value: Any) -> float | None:
    """Normalize a similarity score into the [0.0, 1.0] range when available."""
    if value is None:
        return None
    score = _to_float(value)
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _to_embedding(value: Any) -> list[float] | None:
    """Convert a raw embedding payload to a float list when available."""
    if value is None:
        return None

    if isinstance(value, dict):
        if not value:
            return None
        first = next(iter(value.values()))
        return _to_embedding(first)

    try:
        return [float(component) for component in value]
    except (TypeError, ValueError):
        return None


async def _maybe_await(value: Any) -> Any:
    """Await async values and return sync values unchanged."""
    if inspect.isawaitable(value):
        return await value
    return value


class QdrantAdapter(VectorStoreBackend):
    """Qdrant-backed vector store adapter."""

    def __init__(
        self,
        *,
        qdrant_url: str,
        collection_name: str = "memory_vault",
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self._qdrant_url = qdrant_url
        self._collection_name = collection_name
        self._api_key = api_key

        self._client = client
        self._owns_client = client is None
        self._initialized = False

        self._collection_ready = False
        self._vector_size: int | None = None
        self._models: Any | None = None

    async def initialize(self) -> None:
        """Initialize Qdrant client and detect collection state."""
        if self._initialized:
            return

        if self._client is None:
            try:
                qdrant_client_module = importlib.import_module("qdrant_client")
                self._models = importlib.import_module("qdrant_client.http.models")
            except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency path
                raise ConfigurationError(
                    "QdrantAdapter requires qdrant-client. "
                    "Install memory-vault-ai[qdrant]."
                ) from exc

            async_client = getattr(qdrant_client_module, "AsyncQdrantClient", None)
            if async_client is None:
                raise ConfigurationError("qdrant-client does not expose AsyncQdrantClient")
            self._client = async_client(url=self._qdrant_url, api_key=self._api_key)
        else:
            try:
                self._models = importlib.import_module("qdrant_client.http.models")
            except ModuleNotFoundError:
                self._models = None

        await self._refresh_collection_state()
        self._initialized = True

    async def close(self) -> None:
        """Close adapter resources."""
        client = self._client
        self._client = None
        self._initialized = False
        self._collection_ready = False
        self._vector_size = None

        if not self._owns_client or client is None:
            return

        close_method = getattr(client, "close", None)
        if close_method is None:
            return

        try:
            await _maybe_await(close_method())
        except Exception as exc:  # pragma: no cover - defensive error wrapping
            raise StorageError(f"Failed to close Qdrant client: {exc}") from exc

    async def healthcheck(self) -> dict[str, str]:
        """Return health information for the Qdrant backend."""
        await self.initialize()
        client = self._ensure_client()

        try:
            exists = await self._collection_exists()
            count = 0
            if exists:
                get_collection = getattr(client, "get_collection", None)
                if get_collection is not None:
                    collection_info = await _maybe_await(
                        get_collection(collection_name=self._collection_name)
                    )
                    count = self._extract_points_count(collection_info)
        except Exception as exc:  # pragma: no cover - defensive error wrapping
            raise StorageError(f"Qdrant healthcheck failed: {exc}") from exc

        return {
            "status": "ok",
            "backend": "qdrant",
            "collection": self._collection_name,
            "documents": str(count),
        }

    async def upsert_vectors(self, chunks: Sequence[MemoryChunk]) -> None:
        """Persist vector entries for chunks with embeddings."""
        await self.initialize()
        vector_chunks = [
            chunk
            for chunk in chunks
            if chunk.embedding is not None and chunk.memory_type is not MemoryType.PROCEDURAL
        ]
        if not vector_chunks:
            return

        first_embedding = vector_chunks[0].embedding
        if first_embedding is None:
            return

        vector_size = len(first_embedding)
        await self._ensure_collection(vector_size=vector_size)

        points: list[Any] = []
        for chunk in vector_chunks:
            embedding = chunk.embedding
            if embedding is None:
                continue
            if len(embedding) != vector_size:
                raise StorageError(
                    "Failed to upsert vectors into Qdrant: embedding size mismatch"
                )

            payload = {
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
                "content": chunk.content,
            }
            points.append(self._point(memory_id=chunk.id, embedding=embedding, payload=payload))

        if not points:
            return

        client = self._ensure_client()
        try:
            await _maybe_await(
                client.upsert(
                    collection_name=self._collection_name,
                    points=points,
                    wait=True,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive error wrapping
            raise StorageError(f"Failed to upsert vectors into Qdrant: {exc}") from exc

    async def query_vectors(self, query: MemorySearchQuery) -> list[MemoryChunk]:
        """Run a user-scoped vector query and return mapped memory chunks."""
        await self.initialize()
        if not await self._collection_exists():
            return []

        client = self._ensure_client()
        limit = max(query.top_k * 4, query.top_k)

        try:
            raw_hits = await _maybe_await(
                client.search(
                    collection_name=self._collection_name,
                    query_vector=list(query.query_embedding),
                    query_filter=self._user_filter(query.user_id),
                    limit=limit,
                    with_payload=True,
                    with_vectors=True,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive error wrapping
            raise StorageError(f"Failed to query vectors from Qdrant: {exc}") from exc

        candidates: list[MemoryChunk] = []
        for hit in raw_hits:
            payload = self._payload_from_hit(hit)
            content = str(payload.get("content", "")).strip()
            if not content:
                continue

            memory_type_raw = str(payload.get("memory_type", MemoryType.EPISODIC.value))
            try:
                memory_type = MemoryType(memory_type_raw)
            except ValueError:
                memory_type = MemoryType.EPISODIC

            raw_metadata = payload.get("metadata_json", "{}")
            parsed_metadata: dict[str, Any]
            if isinstance(raw_metadata, str):
                try:
                    loaded = json.loads(raw_metadata)
                    parsed_metadata = loaded if isinstance(loaded, dict) else {}
                except json.JSONDecodeError:
                    parsed_metadata = {}
            elif isinstance(raw_metadata, dict):
                parsed_metadata = raw_metadata
            else:
                parsed_metadata = {}

            chunk = MemoryChunk(
                id=self._point_id(hit),
                user_id=str(payload.get("user_id", query.user_id)),
                session_id=str(payload.get("session_id", "unknown_session")),
                memory_type=memory_type,
                content=content,
                importance=_to_float(payload.get("importance"), default=0.0),
                token_count=_to_int(payload.get("token_count"), default=0),
                embedding=_to_embedding(self._vector_from_hit(hit)),
                compressed=_to_bool(payload.get("compressed"), default=False),
                compression_source=_to_bool(payload.get("compression_source"), default=False),
                source_session_id=(
                    str(payload.get("source_session_id"))
                    if payload.get("source_session_id")
                    else None
                ),
                created_at=_to_utc_datetime(payload.get("created_at")),
                updated_at=_to_utc_datetime(payload.get("updated_at")),
                metadata=parsed_metadata,
                relevance_score=_to_score(self._score_from_hit(hit)),
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
        if not memory_ids:
            return 0
        if not await self._collection_exists():
            return 0

        allowed = await self._find_user_point_ids(
            user_id=user_id,
            include_ids={str(memory_id) for memory_id in memory_ids},
        )
        if not allowed:
            return 0

        await self._delete_point_ids(allowed)
        return len(allowed)

    async def delete_vectors_for_user(self, *, user_id: str) -> int:
        """Delete all vectors for the given user."""
        await self.initialize()
        if not await self._collection_exists():
            return 0

        ids_to_delete = await self._find_user_point_ids(user_id=user_id, include_ids=None)
        if not ids_to_delete:
            return 0

        await self._delete_point_ids(ids_to_delete)
        return len(ids_to_delete)

    def _ensure_client(self) -> Any:
        """Return active client or raise a storage error."""
        if self._client is None:
            raise StorageError("Qdrant client is not initialized")
        return self._client

    async def _refresh_collection_state(self) -> None:
        """Refresh known collection existence and vector size metadata."""
        self._collection_ready = await self._collection_exists()

    async def _collection_exists(self) -> bool:
        """Check whether the target collection already exists."""
        client = self._ensure_client()

        exists_method = getattr(client, "collection_exists", None)
        if exists_method is not None:
            exists_raw = await _maybe_await(
                exists_method(collection_name=self._collection_name)
            )
            exists = bool(exists_raw)
            if exists:
                await self._sync_vector_size()
            return exists

        list_method = getattr(client, "get_collections", None)
        if list_method is not None:
            collections_raw = await _maybe_await(list_method())
            collection_names = self._collection_names(collections_raw)
            exists = self._collection_name in collection_names
            if exists:
                await self._sync_vector_size()
            return exists

        get_method = getattr(client, "get_collection", None)
        if get_method is None:
            return self._collection_ready

        try:
            collection_info = await _maybe_await(get_method(collection_name=self._collection_name))
        except Exception:
            return False

        self._vector_size = self._extract_vector_size(collection_info)
        return True

    async def _sync_vector_size(self) -> None:
        """Read and cache collection vector size when available."""
        client = self._ensure_client()
        get_method = getattr(client, "get_collection", None)
        if get_method is None:
            return

        try:
            collection_info = await _maybe_await(get_method(collection_name=self._collection_name))
        except Exception:
            return
        self._vector_size = self._extract_vector_size(collection_info)

    async def _ensure_collection(self, *, vector_size: int) -> None:
        """Create collection if missing and ensure vector dimensions match."""
        client = self._ensure_client()
        exists = await self._collection_exists()

        if exists:
            if self._vector_size is not None and self._vector_size != vector_size:
                raise StorageError(
                    "Failed to upsert vectors into Qdrant: collection vector size mismatch"
                )
            self._collection_ready = True
            if self._vector_size is None:
                self._vector_size = vector_size
            return

        try:
            if self._models is not None:
                vectors_config = self._models.VectorParams(
                    size=vector_size,
                    distance=self._models.Distance.COSINE,
                )
                await _maybe_await(
                    client.create_collection(
                        collection_name=self._collection_name,
                        vectors_config=vectors_config,
                    )
                )
            else:
                await _maybe_await(
                    client.create_collection(
                        collection_name=self._collection_name,
                        vector_size=vector_size,
                    )
                )
        except Exception as exc:  # pragma: no cover - defensive error wrapping
            raise StorageError(f"Failed to initialize Qdrant collection: {exc}") from exc

        self._collection_ready = True
        self._vector_size = vector_size

    def _point(self, *, memory_id: str, embedding: list[float], payload: dict[str, Any]) -> Any:
        """Build one Qdrant point payload."""
        if self._models is not None:
            return self._models.PointStruct(id=memory_id, vector=embedding, payload=payload)
        return {
            "id": memory_id,
            "vector": embedding,
            "payload": payload,
        }

    def _user_filter(self, user_id: str) -> Any:
        """Build backend filter that constrains operations to one user."""
        if self._models is not None:
            return self._models.Filter(
                must=[
                    self._models.FieldCondition(
                        key="user_id",
                        match=self._models.MatchValue(value=user_id),
                    )
                ]
            )
        return {"user_id": user_id}

    async def _find_user_point_ids(
        self,
        *,
        user_id: str,
        include_ids: set[str] | None,
    ) -> list[str]:
        """Find point IDs belonging to one user, optionally intersected with target IDs."""
        client = self._ensure_client()
        offset: Any | None = None
        discovered: list[str] = []

        while True:
            scroll_result = await _maybe_await(
                client.scroll(
                    collection_name=self._collection_name,
                    scroll_filter=self._user_filter(user_id),
                    limit=256,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False,
                )
            )
            points, next_offset = self._parse_scroll_result(scroll_result)

            for point in points:
                point_id = self._point_id(point)
                if include_ids is not None and point_id not in include_ids:
                    continue
                discovered.append(point_id)

            if next_offset is None:
                break
            offset = next_offset

        return discovered

    async def _delete_point_ids(self, point_ids: Sequence[str]) -> None:
        """Delete exact point IDs from the target collection."""
        if not point_ids:
            return

        client = self._ensure_client()
        try:
            if self._models is not None:
                selector = self._models.PointIdsList(points=list(point_ids))
                await _maybe_await(
                    client.delete(
                        collection_name=self._collection_name,
                        points_selector=selector,
                        wait=True,
                    )
                )
            else:
                await _maybe_await(
                    client.delete(
                        collection_name=self._collection_name,
                        points=list(point_ids),
                        wait=True,
                    )
                )
        except Exception as exc:  # pragma: no cover - defensive error wrapping
            raise StorageError(f"Failed to delete vectors from Qdrant: {exc}") from exc

    @staticmethod
    def _collection_names(collections_response: Any) -> set[str]:
        """Extract collection names from get_collections response payloads."""
        if collections_response is None:
            return set()

        if isinstance(collections_response, dict):
            raw = collections_response.get("collections", [])
        else:
            raw = getattr(collections_response, "collections", [])

        names: set[str] = set()
        for item in raw:
            if isinstance(item, dict):
                name = item.get("name")
            else:
                name = getattr(item, "name", None)
            if name:
                names.add(str(name))
        return names

    @staticmethod
    def _extract_vector_size(collection_info: Any) -> int | None:
        """Extract vector size from collection info payload when present."""
        if collection_info is None:
            return None

        if isinstance(collection_info, dict):
            config = collection_info.get("config", {})
            params = config.get("params", {}) if isinstance(config, dict) else {}
            vectors = params.get("vectors") if isinstance(params, dict) else None
            if isinstance(vectors, dict):
                size = vectors.get("size")
                if size is not None:
                    return _to_int(size, default=0) or None
            return None

        config = getattr(collection_info, "config", None)
        params = getattr(config, "params", None) if config is not None else None
        vectors = getattr(params, "vectors", None) if params is not None else None

        if vectors is None:
            return None
        if hasattr(vectors, "size"):
            return _to_int(vectors.size, default=0) or None
        if isinstance(vectors, dict):
            return _to_int(vectors.get("size"), default=0) or None
        return None

    @staticmethod
    def _extract_points_count(collection_info: Any) -> int:
        """Extract point/document count from collection info payload."""
        if collection_info is None:
            return 0
        if isinstance(collection_info, dict):
            return _to_int(collection_info.get("points_count"), default=0)
        return _to_int(getattr(collection_info, "points_count", 0), default=0)

    @staticmethod
    def _parse_scroll_result(scroll_result: Any) -> tuple[list[Any], Any | None]:
        """Normalize Qdrant scroll result shape to (points, next_offset)."""
        if isinstance(scroll_result, tuple) and len(scroll_result) == 2:
            points_raw, next_offset = scroll_result
            points = list(points_raw) if points_raw is not None else []
            return points, next_offset

        if isinstance(scroll_result, dict):
            points_raw = scroll_result.get("points", [])
            next_offset = scroll_result.get("next_page_offset")
            return list(points_raw), next_offset

        return [], None

    @staticmethod
    def _point_id(point: Any) -> str:
        """Extract canonical string point ID from point/search-hit objects."""
        if isinstance(point, dict):
            return str(point.get("id", ""))
        return str(getattr(point, "id", ""))

    @staticmethod
    def _payload_from_hit(hit: Any) -> dict[str, Any]:
        """Extract payload dictionary from one search hit."""
        if isinstance(hit, dict):
            payload = hit.get("payload", {})
            return payload if isinstance(payload, dict) else {}

        payload = getattr(hit, "payload", {})
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _vector_from_hit(hit: Any) -> Any:
        """Extract raw vector payload from one search hit."""
        if isinstance(hit, dict):
            return hit.get("vector")
        return getattr(hit, "vector", None)

    @staticmethod
    def _score_from_hit(hit: Any) -> Any:
        """Extract raw score payload from one search hit."""
        if isinstance(hit, dict):
            return hit.get("score")
        return getattr(hit, "score", None)


__all__ = ["QdrantAdapter"]
