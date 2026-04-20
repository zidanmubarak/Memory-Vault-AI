from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import anyio

from memory_vault.exceptions import StorageError
from memory_vault.models import MemoryChunk, MemoryType, PaginatedResult
from memory_vault.storage.base import (
    MemoryListQuery,
    MetadataStoreBackend,
    ProceduralMemoryRecord,
    SessionStatsRecord,
)

T = TypeVar("T")


def _to_utc_datetime(value: Any) -> datetime:
    """Convert an arbitrary timestamp value into a UTC datetime."""
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).astimezone(UTC)
    return datetime.now(UTC)


def _to_iso_timestamp(value: datetime) -> str:
    """Convert datetime to normalized ISO-8601 UTC string."""
    return value.astimezone(UTC).isoformat()


def _to_bool(value: Any) -> bool:
    """Convert SQLite-friendly values into bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes"}:
            return True
    return False


def _procedural_id(user_id: str, key: str) -> str:
    """Build a stable procedural memory ID."""
    digest = hashlib.sha1(f"{user_id}:{key}".encode()).hexdigest()[:12]
    return f"proc_{digest}"


class SQLiteAdapter(MetadataStoreBackend):
    """SQLite metadata adapter for chunks, procedural state, and sessions."""

    def __init__(
        self,
        *,
        sqlite_path: str,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        self._sqlite_path = sqlite_path
        self._connection = connection
        self._owns_connection = connection is None
        self._lock = threading.Lock()
        self._initialized = connection is not None

        if self._connection is not None:
            self._connection.row_factory = sqlite3.Row

    async def initialize(self) -> None:
        """Initialize SQLite database and ensure schema exists."""
        if self._initialized:
            return

        if self._connection is None:
            db_path = Path(self._sqlite_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(
                db_path,
                check_same_thread=False,
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
            self._connection.row_factory = sqlite3.Row

        await self._run_sync(self._create_schema)
        self._initialized = True

    async def close(self) -> None:
        """Close SQLite connection if this adapter owns it."""
        connection = self._connection
        self._connection = None
        self._initialized = False

        if connection is not None and self._owns_connection:
            await self._run_sync(connection.close)

    async def healthcheck(self) -> dict[str, str]:
        """Return health information for the SQLite backend."""
        await self.initialize()

        def operation() -> int:
            connection = self._ensure_connection()
            with self._lock:
                cursor = connection.execute("SELECT 1")
                row = cursor.fetchone()
                return int(row[0]) if row is not None else 0

        value = await self._run_sync(operation)
        if value != 1:
            raise StorageError("SQLite healthcheck failed")

        return {
            "status": "ok",
            "backend": "sqlite",
            "path": self._sqlite_path,
        }

    async def upsert_memory_chunks(self, chunks: Sequence[MemoryChunk]) -> list[MemoryChunk]:
        """Persist or update memory chunk metadata."""
        await self.initialize()
        if not chunks:
            return []

        def operation() -> list[MemoryChunk]:
            connection = self._ensure_connection()
            with self._lock:
                cursor = connection.cursor()
                for chunk in chunks:
                    cursor.execute(
                        """
                        INSERT INTO memory_chunks (
                            id,
                            user_id,
                            session_id,
                            memory_type,
                            content,
                            importance,
                            token_count,
                            compressed,
                            compression_source,
                            source_session_id,
                            chroma_id,
                            created_at,
                            updated_at,
                            metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            user_id=excluded.user_id,
                            session_id=excluded.session_id,
                            memory_type=excluded.memory_type,
                            content=excluded.content,
                            importance=excluded.importance,
                            token_count=excluded.token_count,
                            compressed=excluded.compressed,
                            compression_source=excluded.compression_source,
                            source_session_id=excluded.source_session_id,
                            chroma_id=excluded.chroma_id,
                            created_at=excluded.created_at,
                            updated_at=excluded.updated_at,
                            metadata=excluded.metadata
                        """,
                        (
                            chunk.id,
                            chunk.user_id,
                            chunk.session_id,
                            chunk.memory_type.value,
                            chunk.content,
                            chunk.importance,
                            chunk.token_count,
                            int(chunk.compressed),
                            int(chunk.compression_source),
                            chunk.source_session_id,
                            chunk.chroma_id,
                            _to_iso_timestamp(chunk.created_at),
                            _to_iso_timestamp(chunk.updated_at),
                            json.dumps(chunk.metadata, separators=(",", ":")),
                        ),
                    )
                connection.commit()
            return list(chunks)

        return await self._run_sync(operation)

    async def get_memory_chunk(self, *, memory_id: str, user_id: str) -> MemoryChunk | None:
        """Fetch one memory chunk by ID and user scope."""
        await self.initialize()

        def operation() -> MemoryChunk | None:
            connection = self._ensure_connection()
            with self._lock:
                cursor = connection.execute(
                    "SELECT * FROM memory_chunks WHERE id = ? AND user_id = ? LIMIT 1",
                    (memory_id, user_id),
                )
                row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_memory_chunk(row)

        return await self._run_sync(operation)

    async def list_memory_chunks(self, query: MemoryListQuery) -> PaginatedResult[MemoryChunk]:
        """List memory chunks with pagination and filtering."""
        await self.initialize()

        def operation() -> PaginatedResult[MemoryChunk]:
            connection = self._ensure_connection()
            conditions = ["user_id = ?"]
            params: list[Any] = [query.user_id]

            if query.memory_type is not None:
                conditions.append("memory_type = ?")
                params.append(query.memory_type.value)
            if not query.include_compressed:
                conditions.append("compressed = 0")

            where_clause = " AND ".join(conditions)
            offset = (query.page - 1) * query.page_size

            with self._lock:
                count_cursor = connection.execute(
                    f"SELECT COUNT(*) FROM memory_chunks WHERE {where_clause}",
                    params,
                )
                total_row = count_cursor.fetchone()
                total = int(total_row[0]) if total_row is not None else 0

                rows_cursor = connection.execute(
                    f"""
                    SELECT *
                    FROM memory_chunks
                    WHERE {where_clause}
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    [*params, query.page_size, offset],
                )
                rows = rows_cursor.fetchall()

            items = [self._row_to_memory_chunk(row) for row in rows]
            return PaginatedResult[MemoryChunk](
                items=items,
                total=total,
                page=query.page,
                page_size=query.page_size,
            )

        return await self._run_sync(operation)

    async def delete_memory_chunk(self, *, memory_id: str, user_id: str) -> bool:
        """Delete one memory chunk by ID and user scope."""
        await self.initialize()

        def operation() -> bool:
            connection = self._ensure_connection()
            with self._lock:
                cursor = connection.execute(
                    "DELETE FROM memory_chunks WHERE id = ? AND user_id = ?",
                    (memory_id, user_id),
                )
                connection.commit()
            return cursor.rowcount > 0

        return await self._run_sync(operation)

    async def delete_memory_chunks_for_user(self, *, user_id: str) -> int:
        """Delete all memory chunks for a user."""
        await self.initialize()

        def operation() -> int:
            connection = self._ensure_connection()
            with self._lock:
                cursor = connection.execute(
                    "DELETE FROM memory_chunks WHERE user_id = ?",
                    (user_id,),
                )
                connection.commit()
            return int(cursor.rowcount)

        return await self._run_sync(operation)

    async def upsert_procedural_memory(
        self,
        record: ProceduralMemoryRecord,
    ) -> ProceduralMemoryRecord:
        """Persist or update one procedural memory entry."""
        await self.initialize()

        def operation() -> ProceduralMemoryRecord:
            connection = self._ensure_connection()
            with self._lock:
                connection.execute(
                    """
                    INSERT INTO procedural_memory (
                        id,
                        user_id,
                        key,
                        value,
                        confidence,
                        updated_at,
                        source_chunk_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, key) DO UPDATE SET
                        value=excluded.value,
                        confidence=excluded.confidence,
                        updated_at=excluded.updated_at,
                        source_chunk_id=excluded.source_chunk_id
                    """,
                    (
                        _procedural_id(record.user_id, record.key),
                        record.user_id,
                        record.key,
                        record.value,
                        record.confidence,
                        _to_iso_timestamp(record.updated_at),
                        record.source_chunk_id,
                    ),
                )
                connection.commit()
            return record

        return await self._run_sync(operation)

    async def list_procedural_memory(self, *, user_id: str) -> list[ProceduralMemoryRecord]:
        """List all procedural memory entries for a user."""
        await self.initialize()

        def operation() -> list[ProceduralMemoryRecord]:
            connection = self._ensure_connection()
            with self._lock:
                cursor = connection.execute(
                    """
                    SELECT user_id, key, value, confidence, updated_at, source_chunk_id
                    FROM procedural_memory
                    WHERE user_id = ?
                    ORDER BY key ASC
                    """,
                    (user_id,),
                )
                rows = cursor.fetchall()

            return [
                ProceduralMemoryRecord(
                    user_id=str(row["user_id"]),
                    key=str(row["key"]),
                    value=str(row["value"]),
                    confidence=float(row["confidence"]),
                    updated_at=_to_utc_datetime(row["updated_at"]),
                    source_chunk_id=(
                        str(row["source_chunk_id"]) if row["source_chunk_id"] is not None else None
                    ),
                )
                for row in rows
            ]

        return await self._run_sync(operation)

    async def delete_procedural_memory(self, *, user_id: str, key: str) -> bool:
        """Delete one procedural memory key for a user."""
        await self.initialize()

        def operation() -> bool:
            connection = self._ensure_connection()
            with self._lock:
                cursor = connection.execute(
                    "DELETE FROM procedural_memory WHERE user_id = ? AND key = ?",
                    (user_id, key),
                )
                connection.commit()
            return cursor.rowcount > 0

        return await self._run_sync(operation)

    async def upsert_session_stats(self, record: SessionStatsRecord) -> SessionStatsRecord:
        """Persist or update session statistics."""
        await self.initialize()

        def operation() -> SessionStatsRecord:
            connection = self._ensure_connection()
            with self._lock:
                connection.execute(
                    """
                    INSERT INTO sessions (
                        id,
                        user_id,
                        started_at,
                        last_activity,
                        ended_at,
                        compressed,
                        memory_count,
                        total_tokens
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        user_id=excluded.user_id,
                        started_at=excluded.started_at,
                        last_activity=excluded.last_activity,
                        ended_at=excluded.ended_at,
                        compressed=excluded.compressed,
                        memory_count=excluded.memory_count,
                        total_tokens=excluded.total_tokens
                    """,
                    (
                        record.session_id,
                        record.user_id,
                        _to_iso_timestamp(record.started_at),
                        _to_iso_timestamp(record.last_activity),
                        _to_iso_timestamp(record.ended_at) if record.ended_at else None,
                        int(record.compressed),
                        record.memory_count,
                        record.total_tokens_stored,
                    ),
                )
                connection.commit()
            return record

        return await self._run_sync(operation)

    async def get_session_stats(self, *, session_id: str) -> SessionStatsRecord | None:
        """Fetch session statistics by ID."""
        await self.initialize()

        def operation() -> SessionStatsRecord | None:
            connection = self._ensure_connection()
            with self._lock:
                cursor = connection.execute(
                    """
                    SELECT
                        id,
                        user_id,
                        memory_count,
                        total_tokens,
                        started_at,
                        last_activity,
                        ended_at,
                        compressed
                    FROM sessions
                    WHERE id = ?
                    LIMIT 1
                    """,
                    (session_id,),
                )
                row = cursor.fetchone()
            if row is None:
                return None
            return SessionStatsRecord(
                session_id=str(row["id"]),
                user_id=str(row["user_id"]),
                memory_count=int(row["memory_count"]),
                total_tokens_stored=int(row["total_tokens"]),
                started_at=_to_utc_datetime(row["started_at"]),
                last_activity=_to_utc_datetime(row["last_activity"]),
                ended_at=_to_utc_datetime(row["ended_at"]) if row["ended_at"] else None,
                compressed=_to_bool(row["compressed"]),
            )

        return await self._run_sync(operation)

    def _create_schema(self) -> None:
        """Create required schema tables and indexes if missing."""
        connection = self._ensure_connection()
        with self._lock:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_chunks (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance REAL NOT NULL,
                    token_count INTEGER NOT NULL,
                    compressed INTEGER DEFAULT 0,
                    compression_source INTEGER DEFAULT 0,
                    source_session_id TEXT,
                    chroma_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_memory_user ON memory_chunks(user_id);
                CREATE INDEX IF NOT EXISTS idx_memory_user_type
                    ON memory_chunks(user_id, memory_type);
                CREATE INDEX IF NOT EXISTS idx_memory_session ON memory_chunks(session_id);
                CREATE INDEX IF NOT EXISTS idx_memory_compressed
                    ON memory_chunks(user_id, compressed);

                CREATE TABLE IF NOT EXISTS procedural_memory (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    updated_at TEXT NOT NULL,
                    source_chunk_id TEXT
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_proc_user_key
                    ON procedural_memory(user_id, key);

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    last_activity TEXT NOT NULL,
                    ended_at TEXT,
                    compressed INTEGER DEFAULT 0,
                    memory_count INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
                """
            )
            connection.commit()

    def _ensure_connection(self) -> sqlite3.Connection:
        """Return active connection or raise a storage error."""
        if self._connection is None:
            raise StorageError("SQLite connection is not initialized")
        return self._connection

    async def _run_sync(self, operation: Callable[[], T]) -> T:
        """Run blocking SQLite work on a worker thread."""
        try:
            return await anyio.to_thread.run_sync(operation)
        except StorageError:
            raise
        except Exception as exc:  # pragma: no cover - defensive error wrapping
            raise StorageError(f"SQLite operation failed: {exc}") from exc

    @staticmethod
    def _row_to_memory_chunk(row: sqlite3.Row) -> MemoryChunk:
        """Map a SQLite row to MemoryChunk model."""
        try:
            memory_type = MemoryType(str(row["memory_type"]))
        except ValueError:
            memory_type = MemoryType.EPISODIC

        raw_metadata = row["metadata"]
        metadata: dict[str, Any]
        if isinstance(raw_metadata, str):
            try:
                loaded = json.loads(raw_metadata)
                metadata = loaded if isinstance(loaded, dict) else {}
            except json.JSONDecodeError:
                metadata = {}
        else:
            metadata = {}

        return MemoryChunk(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            session_id=str(row["session_id"]),
            memory_type=memory_type,
            content=str(row["content"]),
            importance=float(row["importance"]),
            token_count=int(row["token_count"]),
            compressed=_to_bool(row["compressed"]),
            compression_source=_to_bool(row["compression_source"]),
            source_session_id=str(row["source_session_id"]) if row["source_session_id"] else None,
            chroma_id=str(row["chroma_id"]) if row["chroma_id"] else None,
            created_at=_to_utc_datetime(row["created_at"]),
            updated_at=_to_utc_datetime(row["updated_at"]),
            metadata=metadata,
        )


__all__ = ["SQLiteAdapter"]
