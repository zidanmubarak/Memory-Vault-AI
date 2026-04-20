from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException, Path, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field

from memory_vault.compression import MemoryCompressor
from memory_vault.storage.base import StorageBackend

router = APIRouter(prefix="/v1", tags=["session"])


class SessionStatsResponse(BaseModel):
    """Response model for session statistics."""

    session_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    memory_count: int = Field(ge=0)
    total_tokens_stored: int = Field(ge=0)
    started_at: datetime
    last_activity: datetime
    ended_at: datetime | None = None
    compressed: bool

    model_config = ConfigDict(extra="forbid")


class SessionCompressionAcceptedResponse(BaseModel):
    """Response model for queued session compression requests."""

    job_id: str = Field(min_length=1)
    status: str = Field(min_length=1)
    message: str = Field(min_length=1)

    model_config = ConfigDict(extra="forbid")


async def _run_session_compression(
    *,
    session_id: str,
    storage: StorageBackend,
    compression_threshold: int,
) -> None:
    """Background task that compresses one session."""
    compressor = MemoryCompressor(
        storage=storage,
        compression_threshold=compression_threshold,
    )
    await compressor.compress_session(session_id)


@router.get("/session/{session_id}/stats", response_model=SessionStatsResponse)
async def get_session_stats(
    request: Request,
    session_id: str = Path(min_length=1),
    user_id: str = Query(min_length=1),
) -> SessionStatsResponse:
    """Get aggregate statistics for one session."""
    storage = request.app.state.storage
    record = await storage.get_session_stats(session_id=session_id)
    if record is None or record.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    return SessionStatsResponse(
        session_id=record.session_id,
        user_id=record.user_id,
        memory_count=record.memory_count,
        total_tokens_stored=record.total_tokens_stored,
        started_at=record.started_at,
        last_activity=record.last_activity,
        ended_at=record.ended_at,
        compressed=record.compressed,
    )


@router.post(
    "/session/{session_id}/compress",
    response_model=SessionCompressionAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def compress_session(
    background_tasks: BackgroundTasks,
    request: Request,
    session_id: str = Path(min_length=1),
    user_id: str = Query(min_length=1),
) -> SessionCompressionAcceptedResponse:
    """Queue a background compression task for one session."""
    settings = request.app.state.settings
    storage = request.app.state.storage

    record = await storage.get_session_stats(session_id=session_id)
    if record is None or record.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    job_id = f"job_{uuid4().hex[:8]}"
    background_tasks.add_task(
        _run_session_compression,
        session_id=session_id,
        storage=storage,
        compression_threshold=settings.compression_threshold,
    )

    return SessionCompressionAcceptedResponse(
        job_id=job_id,
        status="queued",
        message="Compression queued. Check /v1/jobs/{job_id} for status.",
    )


__all__ = ["router"]
