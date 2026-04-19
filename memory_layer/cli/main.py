from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Coroutine, Sequence
from datetime import UTC, datetime
from typing import Annotated, Any, TypeVar

import typer
from rich.console import Console
from rich.table import Table

from memory_layer.compression import CompressionResult
from memory_layer.mcp import TOOL_NAMES, run_mcp_server
from memory_layer.models import MemoryChunk, MemoryType, PaginatedResult, RecallResult
from memory_layer.sdk import MemoryLayer

T = TypeVar("T")

app = typer.Typer(help="Memory Layer AI CLI")
mcp_app = typer.Typer(help="Model Context Protocol (MCP) server commands.")
app.add_typer(mcp_app, name="mcp")
console = Console()


def _run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine in CLI command context."""
    return asyncio.run(coro)


def _to_iso(timestamp: datetime) -> str:
    """Return ISO-8601 UTC timestamp with Z suffix."""
    return timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _preview(content: str, *, limit: int = 80) -> str:
    """Return a one-line preview for table output."""
    collapsed = " ".join(content.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[: limit - 3]}..."


def _chunk_payload(chunk: MemoryChunk) -> dict[str, Any]:
    """Serialize one memory chunk for JSON output."""
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


def _compression_payload(result: CompressionResult) -> dict[str, Any]:
    """Serialize one compression result for JSON output."""
    return {
        "user_id": result.user_id,
        "total_uncompressed_sessions": result.total_uncompressed_sessions,
        "sessions_compressed": result.sessions_compressed,
        "summaries_created": result.summaries_created,
        "memories_marked_compressed": result.memories_marked_compressed,
        "compressed_session_ids": list(result.compressed_session_ids),
    }


async def _list_memories(
    *,
    user_id: str,
    memory_type: MemoryType | None,
    page: int,
    page_size: int,
    include_compressed: bool,
) -> PaginatedResult[MemoryChunk]:
    """Load one page of memories for CLI list command."""
    memory = MemoryLayer(user_id=user_id)
    try:
        return await memory.list(
            memory_type=memory_type,
            page=page,
            page_size=page_size,
            include_compressed=include_compressed,
        )
    finally:
        await memory.close()


async def _search_memories(
    *,
    user_id: str,
    query: str,
    top_k: int,
    token_budget: int,
    memory_types: Sequence[MemoryType] | None,
    include_compressed: bool,
) -> RecallResult:
    """Recall memories for CLI search command."""
    memory = MemoryLayer(user_id=user_id)
    try:
        return await memory.recall(
            query,
            top_k=top_k,
            token_budget=token_budget,
            memory_types=list(memory_types) if memory_types is not None else None,
            include_compressed=include_compressed,
        )
    finally:
        await memory.close()


async def _delete_memory(*, user_id: str, memory_id: str) -> bool:
    """Delete one memory by ID."""
    memory = MemoryLayer(user_id=user_id)
    try:
        return await memory.forget(memory_id=memory_id)
    finally:
        await memory.close()


async def _delete_all_memories(*, user_id: str) -> int:
    """Delete all memories for one user."""
    memory = MemoryLayer(user_id=user_id)
    try:
        return await memory.forget_all(confirm=True)
    finally:
        await memory.close()


async def _memory_stats(*, user_id: str, page_size: int) -> dict[str, Any]:
    """Compute aggregate memory statistics for one user."""
    memory = MemoryLayer(user_id=user_id)
    try:
        all_chunks: list[MemoryChunk] = []
        page = 1
        while True:
            result = await memory.list(
                page=page,
                page_size=page_size,
                include_compressed=True,
            )
            all_chunks.extend(result.items)
            if page >= result.total_pages:
                break
            page += 1
    finally:
        await memory.close()

    type_counts = Counter(chunk.memory_type.value for chunk in all_chunks)
    session_count = len({chunk.session_id for chunk in all_chunks})
    compressed_count = sum(1 for chunk in all_chunks if chunk.compressed)

    if all_chunks:
        oldest = min(chunk.created_at for chunk in all_chunks)
        newest = max(chunk.created_at for chunk in all_chunks)
        oldest_value = _to_iso(oldest)
        newest_value = _to_iso(newest)
    else:
        oldest_value = None
        newest_value = None

    return {
        "user_id": user_id,
        "total_memories": len(all_chunks),
        "sessions": session_count,
        "compressed_memories": compressed_count,
        "memory_by_type": {
            memory_type.value: type_counts.get(memory_type.value, 0)
            for memory_type in MemoryType
        },
        "oldest_memory_at": oldest_value,
        "newest_memory_at": newest_value,
    }


async def _compress_memories(
    *,
    user_id: str,
    force: bool,
    sessions_to_compress: int | None,
) -> CompressionResult:
    """Run memory compression for one user."""
    memory = MemoryLayer(user_id=user_id)
    try:
        return await memory.compress(
            force=force,
            sessions_to_compress=sessions_to_compress,
        )
    finally:
        await memory.close()


@app.command("list")
def list_command(
    user_id: Annotated[str, typer.Option("--user-id", help="User identifier.")],
    memory_type: Annotated[
        MemoryType | None,
        typer.Option("--memory-type", help="Filter by memory type."),
    ] = None,
    page: Annotated[int, typer.Option("--page", min=1)] = 1,
    page_size: Annotated[int, typer.Option("--page-size", min=1)] = 20,
    include_compressed: Annotated[bool, typer.Option("--include-compressed")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Output JSON.")] = False,
) -> None:
    """List memories for a user."""
    result = _run_async(
        _list_memories(
            user_id=user_id,
            memory_type=memory_type,
            page=page,
            page_size=page_size,
            include_compressed=include_compressed,
        )
    )

    if as_json:
        payload = {
            "total": result.total,
            "page": result.page,
            "page_size": result.page_size,
            "total_pages": result.total_pages,
            "items": [_chunk_payload(chunk) for chunk in result.items],
        }
        typer.echo(json.dumps(payload, indent=2))
        return

    if not result.items:
        console.print("No memories found.")
        return

    table = Table(title=f"Memories for {user_id}")
    table.add_column("ID")
    table.add_column("Type")
    table.add_column("Importance", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Created")
    table.add_column("Content")

    for chunk in result.items:
        table.add_row(
            chunk.id,
            chunk.memory_type.value,
            f"{chunk.importance:.2f}",
            str(chunk.token_count),
            _to_iso(chunk.created_at),
            _preview(chunk.content),
        )

    console.print(table)
    console.print(
        f"Total: {result.total} | Page: {result.page}/{max(result.total_pages, 1)}"
    )


@app.command("search")
def search_command(
    query: Annotated[str, typer.Argument(help="Query text used for memory recall.")],
    user_id: Annotated[str, typer.Option("--user-id", help="User identifier.")],
    top_k: Annotated[int, typer.Option("--top-k", min=1)] = 5,
    token_budget: Annotated[int, typer.Option("--token-budget", min=1)] = 2000,
    memory_type: Annotated[
        list[MemoryType] | None,
        typer.Option(
            "--memory-type",
            help="Filter by type; repeat the option to include multiple types.",
        ),
    ] = None,
    include_compressed: Annotated[bool, typer.Option("--include-compressed")] = False,
    show_prompt: Annotated[bool, typer.Option("--show-prompt")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Output JSON.")] = False,
) -> None:
    """Search relevant memories for a user query."""
    selected_types = list(memory_type) if memory_type else None
    result = _run_async(
        _search_memories(
            user_id=user_id,
            query=query,
            top_k=top_k,
            token_budget=token_budget,
            memory_types=selected_types,
            include_compressed=include_compressed,
        )
    )

    if as_json:
        payload = {
            "memories": [_chunk_payload(chunk) for chunk in result.memories],
            "total_tokens": result.total_tokens,
            "budget_used": result.budget_used,
            "prompt_block": result.prompt_block,
        }
        typer.echo(json.dumps(payload, indent=2))
        return

    if not result.memories:
        console.print("No memories matched this query.")
        return

    table = Table(title=f"Search results for {user_id}")
    table.add_column("ID")
    table.add_column("Type")
    table.add_column("Relevance", justify="right")
    table.add_column("Importance", justify="right")
    table.add_column("Content")

    for chunk in result.memories:
        relevance = 0.0 if chunk.relevance_score is None else chunk.relevance_score
        table.add_row(
            chunk.id,
            chunk.memory_type.value,
            f"{relevance:.2f}",
            f"{chunk.importance:.2f}",
            _preview(chunk.content),
        )

    console.print(table)
    console.print(
        f"Tokens used: {result.total_tokens} | Budget used: {result.budget_used:.2%}"
    )
    if show_prompt:
        console.print(result.prompt_block)


@app.command("delete")
def delete_command(
    user_id: Annotated[str, typer.Option("--user-id", help="User identifier.")],
    memory_id: Annotated[
        str | None,
        typer.Option("--memory-id", help="Memory identifier."),
    ] = None,
    delete_all: Annotated[
        bool,
        typer.Option("--all", help="Delete all memories for user."),
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", help="Confirm destructive action.")] = False,
) -> None:
    """Delete one memory or all memories for a user."""
    if delete_all and memory_id is not None:
        raise typer.BadParameter("Use either --all or --memory-id, not both.")

    if delete_all:
        if not yes:
            raise typer.BadParameter("--yes is required when using --all.")
        deleted = _run_async(_delete_all_memories(user_id=user_id))
        console.print(f"Deleted {deleted} memory records for user {user_id}.")
        return

    if memory_id is None:
        raise typer.BadParameter("Provide --memory-id, or use --all --yes.")

    deleted = _run_async(_delete_memory(user_id=user_id, memory_id=memory_id))
    if deleted:
        console.print(f"Deleted memory {memory_id}.")
    else:
        console.print(f"Memory {memory_id} was not found for user {user_id}.")


@app.command("stats")
def stats_command(
    user_id: Annotated[str, typer.Option("--user-id", help="User identifier.")],
    page_size: Annotated[int, typer.Option("--page-size", min=1)] = 200,
    as_json: Annotated[bool, typer.Option("--json", help="Output JSON.")] = False,
) -> None:
    """Show aggregate memory statistics for a user."""
    stats = _run_async(_memory_stats(user_id=user_id, page_size=page_size))

    if as_json:
        typer.echo(json.dumps(stats, indent=2))
        return

    table = Table(title=f"Memory stats for {user_id}")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Total memories", str(stats["total_memories"]))
    table.add_row("Sessions", str(stats["sessions"]))
    table.add_row("Compressed memories", str(stats["compressed_memories"]))
    table.add_row("Oldest memory", str(stats["oldest_memory_at"]))
    table.add_row("Newest memory", str(stats["newest_memory_at"]))

    for memory_type in MemoryType:
        type_count = stats["memory_by_type"][memory_type.value]
        table.add_row(f"Type: {memory_type.value}", str(type_count))

    console.print(table)


@app.command("compress")
def compress_command(
    user_id: Annotated[str, typer.Option("--user-id", help="User identifier.")],
    force: Annotated[
        bool,
        typer.Option("--force", help="Compress even when threshold is not exceeded."),
    ] = False,
    sessions_to_compress: Annotated[
        int | None,
        typer.Option("--sessions-to-compress", min=1),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Output JSON.")] = False,
) -> None:
    """Compress old episodic sessions for a user."""
    result = _run_async(
        _compress_memories(
            user_id=user_id,
            force=force,
            sessions_to_compress=sessions_to_compress,
        )
    )

    if as_json:
        typer.echo(json.dumps(_compression_payload(result), indent=2))
        return

    table = Table(title=f"Compression result for {user_id}")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row(
        "Total uncompressed sessions",
        str(result.total_uncompressed_sessions),
    )
    table.add_row("Sessions compressed", str(result.sessions_compressed))
    table.add_row("Summaries created", str(result.summaries_created))
    table.add_row(
        "Memories marked compressed",
        str(result.memories_marked_compressed),
    )
    table.add_row(
        "Compressed sessions",
        ", ".join(result.compressed_session_ids) if result.compressed_session_ids else "-",
    )

    console.print(table)


@mcp_app.command("tools")
def mcp_tools_command(
    as_json: Annotated[bool, typer.Option("--json", help="Output JSON.")] = False,
) -> None:
    """List available MCP tool names."""
    tools = list(TOOL_NAMES)

    if as_json:
        typer.echo(json.dumps({"tools": tools}, indent=2))
        return

    table = Table(title="MCP tools")
    table.add_column("Tool")
    for tool_name in tools:
        table.add_row(tool_name)
    console.print(table)


@mcp_app.command("start")
def mcp_start_command(
    host: Annotated[str, typer.Option("--host", help="Bind host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", min=1, max=65535, help="Bind port.")] = 8001,
) -> None:
    """Start the MCP server."""
    try:
        run_mcp_server(host=host, port=port)
    except RuntimeError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1) from exc


def main() -> None:
    """Run the Memory Layer CLI app."""
    app()


__all__ = ["app", "main"]
