from __future__ import annotations

from time import monotonic
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Path, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field

from memory_layer.api.metrics import ApiMetrics
from memory_layer.models import (
    MemoryChunk,
    MemorySummary,
    MemoryType,
    PaginatedResult,
    RecallResult,
    SaveResult,
)
from memory_layer.sdk import MemoryLayer

router = APIRouter(prefix="/v1", tags=["memory"])


class SaveMemoryRequest(BaseModel):
    """Payload for storing user text in memory."""

    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    memory_type_hint: MemoryType | None = None

    model_config = ConfigDict(extra="forbid")


class DeleteMemoryResponse(BaseModel):
    """Payload returned when deleting one memory by ID."""

    deleted: bool
    id: str = Field(min_length=1)

    model_config = ConfigDict(extra="forbid")


class DeleteAllMemoryRequest(BaseModel):
    """Payload for deleting all memories for a user."""

    user_id: str = Field(min_length=1)
    confirm: Literal[True]

    model_config = ConfigDict(extra="forbid")


class DeleteAllMemoryResponse(BaseModel):
    """Payload returned when deleting all memories for a user."""

    deleted_count: int = Field(ge=0)

    model_config = ConfigDict(extra="forbid")


def _get_metrics(request: Request) -> ApiMetrics | None:
    """Return active API metrics collector when metrics are enabled."""
    metrics = getattr(request.app.state, "metrics", None)
    if isinstance(metrics, ApiMetrics):
        return metrics
    return None


def _parse_memory_types(raw: str | None) -> list[MemoryType] | None:
    """Parse comma-separated memory type query parameter."""
    if raw is None:
        return None

    values = [value.strip().lower() for value in raw.split(",") if value.strip()]
    if not values:
        return None

    parsed: list[MemoryType] = []
    invalid: list[str] = []
    for value in values:
        try:
            parsed.append(MemoryType(value))
        except ValueError:
            invalid.append(value)

    if invalid:
        valid = ", ".join(memory_type.value for memory_type in MemoryType)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "Invalid memory_types value(s): "
                f"{', '.join(invalid)}. Valid values: {valid}."
            ),
        )

    return parsed


@router.post("/memory", response_model=SaveResult, status_code=status.HTTP_201_CREATED)
async def save_memory(payload: SaveMemoryRequest, request: Request) -> SaveResult:
    """Save memory content for a user/session pair."""
    settings = request.app.state.settings
    storage = request.app.state.storage

    sdk = MemoryLayer(
        user_id=payload.user_id,
        session_id=payload.session_id,
        config=settings.to_memory_config(),
        storage=storage,
    )

    start_time = monotonic()
    saved_chunks = await sdk.save(
        payload.text,
        memory_type_hint=payload.memory_type_hint,
        session_id=payload.session_id,
    )

    metrics = _get_metrics(request)
    if metrics is not None:
        metrics.observe_ingestion_latency(monotonic() - start_time)
        for chunk in saved_chunks:
            metrics.increment_memories_total(
                user_id=payload.user_id,
                memory_type=chunk.memory_type.value,
            )

    summaries = [
        MemorySummary(
            id=chunk.id,
            memory_type=chunk.memory_type,
            importance=chunk.importance,
            token_count=chunk.token_count,
            created_at=chunk.created_at,
        )
        for chunk in saved_chunks
    ]

    return SaveResult(saved=summaries, discarded_count=0)


@router.get("/memory/recall", response_model=RecallResult)
async def recall_memory(
    request: Request,
    user_id: str = Query(min_length=1),
    query: str = Query(min_length=1),
    top_k: int = Query(default=5, gt=0),
    token_budget: int = Query(default=2000, gt=0),
    memory_types: str | None = Query(default=None),
) -> RecallResult:
    """Recall relevant memories for a user query."""
    parsed_types = _parse_memory_types(memory_types)
    settings = request.app.state.settings
    storage = request.app.state.storage

    sdk = MemoryLayer(
        user_id=user_id,
        config=settings.to_memory_config(),
        storage=storage,
    )

    start_time = monotonic()
    result = await sdk.recall(
        query,
        top_k=top_k,
        token_budget=token_budget,
        memory_types=parsed_types,
    )

    metrics = _get_metrics(request)
    if metrics is not None:
        metrics.observe_recall_latency(monotonic() - start_time)
        metrics.observe_token_budget_utilization(result.budget_used)

    return result


@router.get("/memory", response_model=PaginatedResult[MemoryChunk])
async def list_memory(
    request: Request,
    user_id: str = Query(min_length=1),
    memory_type: Annotated[MemoryType | None, Query()] = None,
    page: int = Query(default=1, gt=0),
    page_size: int = Query(default=20, gt=0),
    include_compressed: bool = Query(default=False),
) -> PaginatedResult[MemoryChunk]:
    """List memories for one user with pagination and optional filtering."""
    settings = request.app.state.settings
    storage = request.app.state.storage

    sdk = MemoryLayer(
        user_id=user_id,
        config=settings.to_memory_config(),
        storage=storage,
    )

    return await sdk.list(
        memory_type=memory_type,
        page=page,
        page_size=page_size,
        include_compressed=include_compressed,
    )


@router.delete("/memory/{memory_id}", response_model=DeleteMemoryResponse)
async def delete_memory(
    request: Request,
    memory_id: str = Path(min_length=1),
    user_id: str = Query(min_length=1),
) -> DeleteMemoryResponse:
    """Delete one memory by ID for the provided user scope."""
    settings = request.app.state.settings
    storage = request.app.state.storage

    sdk = MemoryLayer(
        user_id=user_id,
        config=settings.to_memory_config(),
        storage=storage,
    )

    deleted = await sdk.forget(memory_id=memory_id)
    return DeleteMemoryResponse(deleted=deleted, id=memory_id)


@router.delete("/memory", response_model=DeleteAllMemoryResponse)
async def delete_all_memory(
    payload: DeleteAllMemoryRequest,
    request: Request,
) -> DeleteAllMemoryResponse:
    """Delete all memories for one user when explicitly confirmed."""
    settings = request.app.state.settings
    storage = request.app.state.storage

    sdk = MemoryLayer(
        user_id=payload.user_id,
        config=settings.to_memory_config(),
        storage=storage,
    )

    deleted_count = await sdk.forget_all(confirm=True)
    return DeleteAllMemoryResponse(deleted_count=deleted_count)


__all__ = ["router"]
