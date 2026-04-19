from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

from memory_layer.exceptions import CompressionError, MemoryLayerError
from memory_layer.models import MemoryChunk, MemoryType
from memory_layer.storage.base import MemoryListQuery, SessionStatsRecord, StorageBackend


def _utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(UTC)


def _whitespace_token_count(text: str) -> int:
    """Return a fallback token estimate based on whitespace splitting."""
    return len(text.split())


class SessionSummarizer(Protocol):
    """Protocol for compression summarizers."""

    async def summarize(
        self,
        *,
        user_id: str,
        session_id: str,
        chunks: Sequence[MemoryChunk],
    ) -> str:
        """Return a semantic summary for one session's episodic chunks."""


class SummaryEmbedder(Protocol):
    """Protocol for optional summary embedding generation."""

    async def encode_batch(self, chunks: list[str]) -> list[list[float]]:
        """Encode summaries into vector embeddings."""


class HeuristicSessionSummarizer:
    """Deterministic fallback summarizer used when no LLM summarizer is configured."""

    _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, *, max_sentences: int = 3, max_chars: int = 900) -> None:
        if max_sentences <= 0:
            raise ValueError("max_sentences must be greater than zero")
        if max_chars <= 0:
            raise ValueError("max_chars must be greater than zero")

        self._max_sentences = max_sentences
        self._max_chars = max_chars

    async def summarize(
        self,
        *,
        user_id: str,
        session_id: str,
        chunks: Sequence[MemoryChunk],
    ) -> str:
        """Build a short session summary by selecting up to three chronological sentences."""
        del user_id, session_id

        ordered_chunks = sorted(chunks, key=lambda chunk: chunk.created_at)
        text_parts = [chunk.content.strip() for chunk in ordered_chunks if chunk.content.strip()]
        if not text_parts:
            return ""

        combined = re.sub(r"\s+", " ", " ".join(text_parts)).strip()
        if not combined:
            return ""

        sentences = [
            sentence.strip()
            for sentence in self._SENTENCE_SPLIT_RE.split(combined)
            if sentence.strip()
        ]
        selected = sentences[: self._max_sentences] or [combined]

        summary = " ".join(selected).strip()
        if len(summary) > self._max_chars:
            summary = summary[: self._max_chars].rstrip()
        if summary and summary[-1].isalnum():
            summary = f"{summary}."
        return summary


@dataclass(frozen=True, slots=True)
class CompressionResult:
    """Result payload for one compression run."""

    user_id: str
    total_uncompressed_sessions: int
    sessions_compressed: int
    summaries_created: int
    memories_marked_compressed: int
    compressed_session_ids: tuple[str, ...] = ()


class MemoryCompressor:
    """Session-based episodic memory compressor with semantic summary creation."""

    def __init__(
        self,
        *,
        storage: StorageBackend,
        summarizer: SessionSummarizer | None = None,
        embedder: SummaryEmbedder | None = None,
        compression_threshold: int = 10,
        sessions_to_compress: int | None = None,
        token_counter: Callable[[str], int] | None = None,
        now_provider: Callable[[], datetime] | None = None,
        page_size: int = 200,
    ) -> None:
        if compression_threshold <= 0:
            raise ValueError("compression_threshold must be greater than zero")
        if sessions_to_compress is not None and sessions_to_compress <= 0:
            raise ValueError("sessions_to_compress must be greater than zero")
        if page_size <= 0:
            raise ValueError("page_size must be greater than zero")

        self._storage = storage
        self._summarizer = summarizer or HeuristicSessionSummarizer()
        self._embedder = embedder
        self._compression_threshold = compression_threshold
        self._default_sessions_to_compress = sessions_to_compress
        self._token_counter = token_counter or _whitespace_token_count
        self._now = now_provider or _utc_now
        self._page_size = page_size

    async def compress_user(
        self,
        user_id: str,
        *,
        force: bool = False,
        sessions_to_compress: int | None = None,
    ) -> CompressionResult:
        """Compress oldest episodic sessions for one user into semantic summaries."""
        if not user_id:
            raise ValueError("user_id is required")
        if sessions_to_compress is not None and sessions_to_compress <= 0:
            raise ValueError("sessions_to_compress must be greater than zero")

        try:
            episodic = await self._load_all_episodic_chunks(user_id=user_id)
            uncompressed = [chunk for chunk in episodic if not chunk.compressed]
            ordered_sessions = self._ordered_sessions(uncompressed)
            total_uncompressed_sessions = len(ordered_sessions)

            if total_uncompressed_sessions == 0:
                return CompressionResult(
                    user_id=user_id,
                    total_uncompressed_sessions=0,
                    sessions_compressed=0,
                    summaries_created=0,
                    memories_marked_compressed=0,
                )

            if not force and total_uncompressed_sessions <= self._compression_threshold:
                return CompressionResult(
                    user_id=user_id,
                    total_uncompressed_sessions=total_uncompressed_sessions,
                    sessions_compressed=0,
                    summaries_created=0,
                    memories_marked_compressed=0,
                )

            target_sessions = self._resolve_target_sessions(
                available=total_uncompressed_sessions,
                explicit=sessions_to_compress,
            )
            selected_sessions = ordered_sessions[:target_sessions]
            return await self._compress_sessions(
                user_id=user_id,
                selected_sessions=selected_sessions,
                total_uncompressed_sessions=total_uncompressed_sessions,
            )
        except MemoryLayerError as exc:
            if isinstance(exc, CompressionError):
                raise
            raise CompressionError(f"Failed to compress memories: {exc}") from exc
        except Exception as exc:
            raise CompressionError(f"Failed to compress memories: {exc}") from exc

    async def compress_session(self, session_id: str) -> CompressionResult:
        """Compress one specific episodic session into a semantic summary."""
        if not session_id:
            raise ValueError("session_id is required")

        try:
            session_stats = await self._storage.get_session_stats(session_id=session_id)
            if session_stats is None:
                return CompressionResult(
                    user_id="",
                    total_uncompressed_sessions=0,
                    sessions_compressed=0,
                    summaries_created=0,
                    memories_marked_compressed=0,
                )

            user_id = session_stats.user_id
            episodic = await self._load_all_episodic_chunks(user_id=user_id)
            session_chunks = [
                chunk
                for chunk in episodic
                if chunk.session_id == session_id and not chunk.compressed
            ]

            selected_sessions: list[tuple[str, list[MemoryChunk]]] = []
            if session_chunks:
                session_chunks.sort(key=lambda chunk: chunk.created_at)
                selected_sessions.append((session_id, session_chunks))

            return await self._compress_sessions(
                user_id=user_id,
                selected_sessions=selected_sessions,
                total_uncompressed_sessions=len(selected_sessions),
            )
        except MemoryLayerError as exc:
            if isinstance(exc, CompressionError):
                raise
            raise CompressionError(f"Failed to compress memories: {exc}") from exc
        except Exception as exc:
            raise CompressionError(f"Failed to compress memories: {exc}") from exc

    async def _compress_sessions(
        self,
        *,
        user_id: str,
        selected_sessions: Sequence[tuple[str, list[MemoryChunk]]],
        total_uncompressed_sessions: int,
    ) -> CompressionResult:
        """Compress selected sessions and persist summaries/archive flags."""
        if not selected_sessions:
            return CompressionResult(
                user_id=user_id,
                total_uncompressed_sessions=total_uncompressed_sessions,
                sessions_compressed=0,
                summaries_created=0,
                memories_marked_compressed=0,
            )

        run_time = self._now()
        summary_chunks: list[MemoryChunk] = []
        marked_compressed: list[MemoryChunk] = []
        compressed_session_ids: list[str] = []

        for session_id, chunks in selected_sessions:
            summary_text = await self._summarizer.summarize(
                user_id=user_id,
                session_id=session_id,
                chunks=chunks,
            )
            normalized_summary = summary_text.strip()
            if not normalized_summary:
                continue

            summary_embedding = await self._encode_summary(normalized_summary)
            summary_chunks.append(
                MemoryChunk(
                    id=self._memory_id(),
                    user_id=user_id,
                    session_id=session_id,
                    memory_type=MemoryType.SEMANTIC,
                    content=normalized_summary,
                    importance=self._summary_importance(chunks),
                    token_count=self._token_count(normalized_summary),
                    embedding=summary_embedding,
                    compressed=False,
                    compression_source=True,
                    source_session_id=session_id,
                    created_at=run_time,
                    updated_at=run_time,
                    metadata={
                        "compression": "session_summary",
                        "source_chunk_ids": [chunk.id for chunk in chunks],
                    },
                )
            )

            for chunk in chunks:
                marked_compressed.append(
                    chunk.model_copy(
                        update={
                            "compressed": True,
                            "updated_at": run_time,
                        }
                    )
                )

            compressed_session_ids.append(session_id)

        if not summary_chunks:
            return CompressionResult(
                user_id=user_id,
                total_uncompressed_sessions=total_uncompressed_sessions,
                sessions_compressed=0,
                summaries_created=0,
                memories_marked_compressed=0,
            )

        await self._storage.upsert_memory_chunks([*summary_chunks, *marked_compressed])
        for session_id in compressed_session_ids:
            await self._mark_session_compressed(session_id=session_id, run_time=run_time)

        return CompressionResult(
            user_id=user_id,
            total_uncompressed_sessions=total_uncompressed_sessions,
            sessions_compressed=len(compressed_session_ids),
            summaries_created=len(summary_chunks),
            memories_marked_compressed=len(marked_compressed),
            compressed_session_ids=tuple(compressed_session_ids),
        )

    async def _load_all_episodic_chunks(self, *, user_id: str) -> list[MemoryChunk]:
        """Load all episodic chunks for one user using pagination."""
        page = 1
        collected: list[MemoryChunk] = []

        while True:
            page_result = await self._storage.list_memory_chunks(
                MemoryListQuery(
                    user_id=user_id,
                    memory_type=MemoryType.EPISODIC,
                    include_compressed=True,
                    page=page,
                    page_size=self._page_size,
                )
            )
            if not page_result.items:
                break

            collected.extend(page_result.items)
            if page >= page_result.total_pages:
                break
            page += 1

        return collected

    @staticmethod
    def _ordered_sessions(chunks: Sequence[MemoryChunk]) -> list[tuple[str, list[MemoryChunk]]]:
        """Group chunks by session and sort sessions by oldest timestamp."""
        grouped: dict[str, list[MemoryChunk]] = {}
        for chunk in chunks:
            grouped.setdefault(chunk.session_id, []).append(chunk)

        for chunk_list in grouped.values():
            chunk_list.sort(key=lambda chunk: chunk.created_at)

        return sorted(grouped.items(), key=lambda item: item[1][0].created_at)

    def _resolve_target_sessions(self, *, available: int, explicit: int | None) -> int:
        """Resolve how many sessions should be compressed this run."""
        if explicit is not None:
            return min(explicit, available)
        if self._default_sessions_to_compress is not None:
            return min(self._default_sessions_to_compress, available)

        default_target = max(1, self._compression_threshold // 2)
        return min(default_target, available)

    async def _encode_summary(self, summary: str) -> list[float] | None:
        """Embed summary text when an embedder is configured."""
        if self._embedder is None:
            return None

        vectors = await self._embedder.encode_batch([summary])
        if not vectors:
            return None
        return vectors[0]

    async def _mark_session_compressed(self, *, session_id: str, run_time: datetime) -> None:
        """Set compressed flag on session metadata when present."""
        existing = await self._storage.get_session_stats(session_id=session_id)
        if existing is None:
            return

        updated_last_activity = (
            run_time if run_time >= existing.last_activity else existing.last_activity
        )
        updated = SessionStatsRecord(
            session_id=existing.session_id,
            user_id=existing.user_id,
            memory_count=existing.memory_count,
            total_tokens_stored=existing.total_tokens_stored,
            started_at=existing.started_at,
            last_activity=updated_last_activity,
            ended_at=existing.ended_at,
            compressed=True,
        )
        await self._storage.upsert_session_stats(updated)

    def _token_count(self, text: str) -> int:
        """Return summary token count with a minimum of one."""
        token_count = self._token_counter(text)
        return token_count if token_count > 0 else 1

    @staticmethod
    def _summary_importance(chunks: Sequence[MemoryChunk]) -> float:
        """Compute summary importance from source chunk importances."""
        if not chunks:
            return 0.0
        average = sum(chunk.importance for chunk in chunks) / len(chunks)
        if average < 0.0:
            return 0.0
        if average > 1.0:
            return 1.0
        return average

    @staticmethod
    def _memory_id() -> str:
        """Generate canonical memory identifier."""
        return f"mem_{uuid4().hex[:12]}"


__all__ = ["CompressionResult", "HeuristicSessionSummarizer", "MemoryCompressor"]

