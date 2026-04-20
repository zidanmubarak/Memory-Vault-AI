from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException, Path, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field

from memory_vault.sdk import MemoryLayer

router = APIRouter(prefix="/v1", tags=["procedural"])


class ProceduralMemoryItemResponse(BaseModel):
    """Response model for one procedural memory item."""

    key: str = Field(min_length=1)
    value: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    updated_at: datetime
    source_chunk_id: str | None = None

    model_config = ConfigDict(extra="forbid")


class ProceduralMemoryListResponse(BaseModel):
    """Response model for listing procedural memory items."""

    items: list[ProceduralMemoryItemResponse] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class UpsertProceduralMemoryRequest(BaseModel):
    """Payload for creating or updating a procedural memory preference."""

    user_id: str = Field(min_length=1)
    key: str = Field(min_length=1)
    value: str = Field(min_length=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_chunk_id: str | None = None

    model_config = ConfigDict(extra="forbid")


class DeleteProceduralMemoryResponse(BaseModel):
    """Response model for procedural memory delete operations."""

    deleted: bool
    key: str = Field(min_length=1)

    model_config = ConfigDict(extra="forbid")


@router.get("/procedural", response_model=ProceduralMemoryListResponse)
async def list_procedural_memory(
    request: Request,
    user_id: str = Query(min_length=1),
) -> ProceduralMemoryListResponse:
    """List procedural memory preferences for one user."""
    settings = request.app.state.settings
    storage = request.app.state.storage

    sdk = MemoryLayer(
        user_id=user_id,
        config=settings.to_memory_config(),
        storage=storage,
    )
    records = await sdk.list_procedural_memory()

    return ProceduralMemoryListResponse(
        items=[
            ProceduralMemoryItemResponse(
                key=record.key,
                value=record.value,
                confidence=record.confidence,
                updated_at=record.updated_at,
                source_chunk_id=record.source_chunk_id,
            )
            for record in records
        ]
    )


@router.put("/procedural", response_model=ProceduralMemoryItemResponse)
async def upsert_procedural_memory(
    payload: UpsertProceduralMemoryRequest,
    request: Request,
) -> ProceduralMemoryItemResponse:
    """Create or update one procedural memory preference for one user."""
    settings = request.app.state.settings
    storage = request.app.state.storage

    sdk = MemoryLayer(
        user_id=payload.user_id,
        config=settings.to_memory_config(),
        storage=storage,
    )
    record = await sdk.upsert_procedural_memory(
        key=payload.key,
        value=payload.value,
        confidence=payload.confidence,
        source_chunk_id=payload.source_chunk_id,
    )

    return ProceduralMemoryItemResponse(
        key=record.key,
        value=record.value,
        confidence=record.confidence,
        updated_at=record.updated_at,
        source_chunk_id=record.source_chunk_id,
    )


@router.delete(
    "/procedural/{key}",
    response_model=DeleteProceduralMemoryResponse,
)
async def delete_procedural_memory(
    request: Request,
    key: str = Path(min_length=1),
    user_id: str = Query(min_length=1),
) -> DeleteProceduralMemoryResponse:
    """Delete one procedural memory preference for one user by key."""
    settings = request.app.state.settings
    storage = request.app.state.storage

    sdk = MemoryLayer(
        user_id=user_id,
        config=settings.to_memory_config(),
        storage=storage,
    )
    deleted = await sdk.delete_procedural_memory(key=key)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Procedural memory not found",
        )

    return DeleteProceduralMemoryResponse(deleted=True, key=key)


__all__ = ["router"]
