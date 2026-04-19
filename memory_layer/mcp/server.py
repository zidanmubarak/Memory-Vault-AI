from __future__ import annotations

import importlib
import json
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from memory_layer import __version__
from memory_layer.config import Settings, get_settings
from memory_layer.models import MemoryChunk, MemoryType
from memory_layer.sdk import MemoryLayer

TOOL_NAMES: tuple[str, ...] = (
    "memory_save",
    "memory_recall",
    "memory_list",
    "memory_forget",
)


class JsonRpcRequest(BaseModel):
    """JSON-RPC 2.0 request payload."""

    jsonrpc: str = Field(default="2.0")
    id: int | str | None = None
    method: str = Field(min_length=1)
    params: dict[str, Any] | None = None

    model_config = ConfigDict(extra="forbid")


class ToolCallParams(BaseModel):
    """Parameters for MCP tools/call requests."""

    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class MemorySaveArgs(BaseModel):
    """Arguments for the memory_save tool."""

    text: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    memory_type: MemoryType | None = None

    model_config = ConfigDict(extra="forbid")


class MemoryRecallArgs(BaseModel):
    """Arguments for the memory_recall tool."""

    query: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    top_k: int = Field(default=5, gt=0)
    token_budget: int = Field(default=2000, gt=0)
    memory_types: list[MemoryType] | None = None
    include_compressed: bool = False

    model_config = ConfigDict(extra="forbid")


class MemoryListArgs(BaseModel):
    """Arguments for the memory_list tool."""

    user_id: str = Field(min_length=1)
    memory_type: MemoryType | None = None
    page: int = Field(default=1, gt=0)
    page_size: int = Field(default=20, gt=0)
    include_compressed: bool = False

    model_config = ConfigDict(extra="forbid")


class MemoryForgetArgs(BaseModel):
    """Arguments for the memory_forget tool."""

    user_id: str = Field(min_length=1)
    memory_id: str | None = None
    confirm: bool = False

    model_config = ConfigDict(extra="forbid")


def _to_iso(timestamp: datetime) -> str:
    """Return ISO-8601 UTC timestamp with Z suffix."""
    return timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_bearer_token(authorization: str | None) -> str | None:
    """Extract bearer token from Authorization header when present."""
    if not authorization:
        return None

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return None

    normalized = token.strip()
    if not normalized:
        return None
    return normalized


def _unauthorized_response() -> JSONResponse:
    """Build a standard unauthorized response payload."""
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={"detail": "Unauthorized"},
        headers={"WWW-Authenticate": "Bearer"},
    )


def _memory_chunk_payload(chunk: MemoryChunk) -> dict[str, Any]:
    """Serialize one memory chunk for tool responses."""
    return {
        "id": chunk.id,
        "user_id": chunk.user_id,
        "session_id": chunk.session_id,
        "memory_type": chunk.memory_type.value,
        "content": chunk.content,
        "importance": chunk.importance,
        "token_count": chunk.token_count,
        "compressed": chunk.compressed,
        "compression_source": chunk.compression_source,
        "source_session_id": chunk.source_session_id,
        "relevance_score": chunk.relevance_score,
        "created_at": _to_iso(chunk.created_at),
        "updated_at": _to_iso(chunk.updated_at),
    }


def _memory_summary_payload(chunk: MemoryChunk) -> dict[str, Any]:
    """Serialize summary memory fields for save responses."""
    return {
        "id": chunk.id,
        "memory_type": chunk.memory_type.value,
        "importance": chunk.importance,
        "token_count": chunk.token_count,
        "created_at": _to_iso(chunk.created_at),
    }


async def _memory_save(arguments: dict[str, Any], *, settings: Settings) -> dict[str, Any]:
    """Handle memory_save tool call."""
    args = MemorySaveArgs.model_validate(arguments)
    memory = MemoryLayer(
        user_id=args.user_id,
        session_id=args.session_id,
        settings=settings,
    )
    try:
        saved = await memory.save(
            args.text,
            memory_type_hint=args.memory_type,
            session_id=args.session_id,
        )
    finally:
        await memory.close()

    return {
        "saved": [_memory_summary_payload(chunk) for chunk in saved],
        "discarded_count": 0,
    }


async def _memory_recall(arguments: dict[str, Any], *, settings: Settings) -> dict[str, Any]:
    """Handle memory_recall tool call."""
    args = MemoryRecallArgs.model_validate(arguments)
    memory = MemoryLayer(user_id=args.user_id, settings=settings)
    try:
        result = await memory.recall(
            args.query,
            top_k=args.top_k,
            token_budget=args.token_budget,
            memory_types=args.memory_types,
            include_compressed=args.include_compressed,
        )
    finally:
        await memory.close()

    return {
        "memories": [_memory_chunk_payload(chunk) for chunk in result.memories],
        "total_tokens": result.total_tokens,
        "budget_used": result.budget_used,
        "prompt_block": result.prompt_block,
    }


async def _memory_list(arguments: dict[str, Any], *, settings: Settings) -> dict[str, Any]:
    """Handle memory_list tool call."""
    args = MemoryListArgs.model_validate(arguments)
    memory = MemoryLayer(user_id=args.user_id, settings=settings)
    try:
        result = await memory.list(
            memory_type=args.memory_type,
            page=args.page,
            page_size=args.page_size,
            include_compressed=args.include_compressed,
        )
    finally:
        await memory.close()

    return {
        "items": [_memory_chunk_payload(chunk) for chunk in result.items],
        "total": result.total,
        "page": result.page,
        "page_size": result.page_size,
        "total_pages": result.total_pages,
    }


async def _memory_forget(arguments: dict[str, Any], *, settings: Settings) -> dict[str, Any]:
    """Handle memory_forget tool call."""
    args = MemoryForgetArgs.model_validate(arguments)
    memory = MemoryLayer(user_id=args.user_id, settings=settings)
    try:
        if args.memory_id is not None:
            deleted = await memory.forget(memory_id=args.memory_id)
            return {
                "deleted": deleted,
                "deleted_count": 1 if deleted else 0,
                "memory_id": args.memory_id,
            }

        if args.confirm:
            deleted_count = await memory.forget_all(confirm=True)
            return {
                "deleted": deleted_count > 0,
                "deleted_count": deleted_count,
            }

        raise ValueError("memory_forget requires memory_id or confirm=true")
    finally:
        await memory.close()


async def _call_tool(name: str, arguments: dict[str, Any], *, settings: Settings) -> dict[str, Any]:
    """Dispatch tool calls to MemoryLayer-backed handlers."""
    if name == "memory_save":
        return await _memory_save(arguments, settings=settings)
    if name == "memory_recall":
        return await _memory_recall(arguments, settings=settings)
    if name == "memory_list":
        return await _memory_list(arguments, settings=settings)
    if name == "memory_forget":
        return await _memory_forget(arguments, settings=settings)

    raise ValueError(f"Unknown tool: {name}")


def get_tool_definitions() -> list[dict[str, Any]]:
    """Return MCP tool definitions with descriptions and input schemas."""
    return [
        {
            "name": "memory_save",
            "description": "Save a memory for a user and session.",
            "inputSchema": MemorySaveArgs.model_json_schema(),
        },
        {
            "name": "memory_recall",
            "description": "Recall memories relevant to a query.",
            "inputSchema": MemoryRecallArgs.model_json_schema(),
        },
        {
            "name": "memory_list",
            "description": "List stored memories for a user.",
            "inputSchema": MemoryListArgs.model_json_schema(),
        },
        {
            "name": "memory_forget",
            "description": "Delete one memory or all memories for a user.",
            "inputSchema": MemoryForgetArgs.model_json_schema(),
        },
    ]


def _jsonrpc_result(*, request_id: int | str | None, result: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON-RPC success payload."""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": result,
    }


def _jsonrpc_error(
    *,
    request_id: int | str | None,
    code: int,
    message: str,
) -> dict[str, Any]:
    """Build a JSON-RPC error payload."""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": code,
            "message": message,
        },
    }


def create_mcp_app(*, settings: Settings | None = None) -> FastAPI:
    """Create FastAPI app exposing MCP-compatible memory tools."""
    resolved_settings = settings or get_settings()

    app = FastAPI(
        title="Memory Layer AI MCP",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next: Any) -> Any:
        """Apply optional bearer auth when ML_API_KEY is configured."""
        if (
            resolved_settings.api_key
            and request.url.path.startswith("/mcp/v1")
            and request.url.path != "/mcp/v1/health"
        ):
            token = _parse_bearer_token(request.headers.get("Authorization"))
            if token != resolved_settings.api_key:
                return _unauthorized_response()

        return await call_next(request)

    @app.get("/mcp/v1/health")
    async def health() -> dict[str, Any]:
        """Return MCP server health status and registered tools."""
        return {
            "status": "ok",
            "version": __version__,
            "tools": list(TOOL_NAMES),
        }

    @app.get("/mcp/v1")
    async def info() -> dict[str, Any]:
        """Return simple endpoint metadata for manual checks."""
        return {
            "server": "memory-layer-ai",
            "endpoint": "/mcp/v1",
            "transport": "json-rpc",
            "tools": list(TOOL_NAMES),
        }

    @app.post("/mcp/v1")
    async def mcp(payload: JsonRpcRequest) -> dict[str, Any]:
        """Handle JSON-RPC MCP requests."""
        request_id = payload.id

        if payload.jsonrpc != "2.0":
            return _jsonrpc_error(
                request_id=request_id,
                code=-32600,
                message="Invalid Request: jsonrpc must be '2.0'",
            )

        if payload.method == "initialize":
            return _jsonrpc_result(
                request_id=request_id,
                result={
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {
                        "name": "memory-layer-ai",
                        "version": __version__,
                    },
                    "capabilities": {
                        "tools": {
                            "listChanged": False,
                        }
                    },
                },
            )

        if payload.method == "tools/list":
            return _jsonrpc_result(
                request_id=request_id,
                result={"tools": get_tool_definitions()},
            )

        if payload.method == "tools/call":
            try:
                params = ToolCallParams.model_validate(payload.params or {})
                tool_result = await _call_tool(
                    params.name,
                    params.arguments,
                    settings=resolved_settings,
                )
            except ValidationError as exc:
                return _jsonrpc_error(
                    request_id=request_id,
                    code=-32602,
                    message=str(exc),
                )
            except ValueError as exc:
                return _jsonrpc_error(
                    request_id=request_id,
                    code=-32602,
                    message=str(exc),
                )
            except Exception as exc:  # pragma: no cover - defensive fallback
                return _jsonrpc_error(
                    request_id=request_id,
                    code=-32000,
                    message=f"Tool call failed: {exc}",
                )

            return _jsonrpc_result(
                request_id=request_id,
                result={
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(tool_result, separators=(",", ":")),
                        }
                    ],
                    "structuredContent": tool_result,
                    "isError": False,
                },
            )

        return _jsonrpc_error(
            request_id=request_id,
            code=-32601,
            message=f"Method not found: {payload.method}",
        )

    return app


def run_mcp_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8001,
    settings: Settings | None = None,
) -> None:
    """Start the MCP server with uvicorn."""
    uvicorn_module: Any
    try:
        uvicorn_module = importlib.import_module("uvicorn")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "uvicorn is required for MCP server startup. "
            "Install with: pip install \"memory-layer-ai[mcp]\""
        ) from exc

    run_callable = getattr(uvicorn_module, "run", None)
    if not callable(run_callable):
        raise RuntimeError("uvicorn.run is not available")

    run_callable(
        create_mcp_app(settings=settings),
        host=host,
        port=port,
    )


__all__ = ["TOOL_NAMES", "create_mcp_app", "get_tool_definitions", "run_mcp_server"]
