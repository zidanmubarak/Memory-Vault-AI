from __future__ import annotations

import importlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, ClassVar

from typer.testing import CliRunner

from memory_vault.compression import CompressionResult
from memory_vault.models import MemoryChunk, MemoryType, PaginatedResult, RecallResult

cli_main = importlib.import_module("memory_vault.cli.main")


class FakeMemoryLayer:
    records: ClassVar[list[MemoryChunk]] = []
    recall_calls: ClassVar[list[dict[str, Any]]] = []
    compress_calls: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, *, user_id: str, **kwargs: Any) -> None:
        del kwargs
        self.user_id = user_id

    async def close(self) -> None:
        return None

    async def list(
        self,
        *,
        memory_type: MemoryType | None = None,
        page: int = 1,
        page_size: int = 20,
        include_compressed: bool = False,
    ) -> PaginatedResult[MemoryChunk]:
        filtered = [chunk for chunk in FakeMemoryLayer.records if chunk.user_id == self.user_id]
        if memory_type is not None:
            filtered = [chunk for chunk in filtered if chunk.memory_type is memory_type]
        if not include_compressed:
            filtered = [chunk for chunk in filtered if not chunk.compressed]

        filtered.sort(key=lambda chunk: chunk.created_at, reverse=True)
        total = len(filtered)
        start = (page - 1) * page_size
        end = start + page_size
        return PaginatedResult[MemoryChunk](
            items=filtered[start:end],
            total=total,
            page=page,
            page_size=page_size,
        )

    async def recall(
        self,
        query: str,
        *,
        top_k: int = 5,
        token_budget: int = 2000,
        memory_types: Sequence[MemoryType] | None = None,
        include_compressed: bool = False,
        reranker_enabled: bool | None = None,
    ) -> RecallResult:
        del query, reranker_enabled
        FakeMemoryLayer.recall_calls.append(
            {
                "top_k": top_k,
                "token_budget": token_budget,
                "memory_types": [*memory_types] if memory_types is not None else [],
                "include_compressed": include_compressed,
            }
        )

        selected = [
            chunk for chunk in FakeMemoryLayer.records if chunk.user_id == self.user_id
        ]
        if memory_types:
            selected = [chunk for chunk in selected if chunk.memory_type in memory_types]
        if not include_compressed:
            selected = [chunk for chunk in selected if not chunk.compressed]

        selected = selected[:top_k]
        tokens = sum(chunk.token_count for chunk in selected)
        budget_used = 0.0 if token_budget == 0 else tokens / token_budget
        return RecallResult(
            memories=selected,
            total_tokens=tokens,
            budget_used=budget_used,
            prompt_block="<memory>\n</memory>",
        )

    async def forget(self, *, memory_id: str) -> bool:
        for index, chunk in enumerate(FakeMemoryLayer.records):
            if chunk.user_id == self.user_id and chunk.id == memory_id:
                FakeMemoryLayer.records.pop(index)
                return True
        return False

    async def forget_all(self, *, confirm: bool = False) -> int:
        if not confirm:
            raise ValueError("confirm must be true")

        before = len(FakeMemoryLayer.records)
        FakeMemoryLayer.records = [
            chunk for chunk in FakeMemoryLayer.records if chunk.user_id != self.user_id
        ]
        return before - len(FakeMemoryLayer.records)

    async def compress(
        self,
        *,
        force: bool = False,
        sessions_to_compress: int | None = None,
    ) -> CompressionResult:
        FakeMemoryLayer.compress_calls.append(
            {
                "force": force,
                "sessions_to_compress": sessions_to_compress,
            }
        )

        compressed_count = 1 if sessions_to_compress is None else sessions_to_compress
        session_ids = tuple(f"sess_{index + 1}" for index in range(compressed_count))
        return CompressionResult(
            user_id=self.user_id,
            total_uncompressed_sessions=3,
            sessions_compressed=compressed_count,
            summaries_created=compressed_count,
            memories_marked_compressed=compressed_count * 2,
            compressed_session_ids=session_ids,
        )


runner = CliRunner()


def _chunk(
    *,
    memory_id: str,
    user_id: str,
    session_id: str,
    memory_type: MemoryType,
    compressed: bool = False,
) -> MemoryChunk:
    now = datetime(2026, 4, 19, 10, 0, tzinfo=UTC)
    return MemoryChunk(
        id=memory_id,
        user_id=user_id,
        session_id=session_id,
        memory_type=memory_type,
        content=f"content for {memory_id}",
        importance=0.8,
        token_count=4,
        embedding=[0.1, 0.2],
        compressed=compressed,
        created_at=now,
        updated_at=now,
    )


def _reset_fake_memory_vault() -> None:
    FakeMemoryLayer.records.clear()
    FakeMemoryLayer.recall_calls.clear()
    FakeMemoryLayer.compress_calls.clear()


def test_list_command_json_output(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _reset_fake_memory_vault()
    FakeMemoryLayer.records.append(
        _chunk(
            memory_id="mem_1",
            user_id="user_a",
            session_id="sess_1",
            memory_type=MemoryType.SEMANTIC,
        )
    )
    monkeypatch.setattr(cli_main, "MemoryLayer", FakeMemoryLayer)

    result = runner.invoke(cli_main.app, ["list", "--user-id", "user_a", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["total"] == 1
    assert payload["items"][0]["id"] == "mem_1"
    assert payload["items"][0]["memory_type"] == "semantic"


def test_search_command_passes_memory_types(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _reset_fake_memory_vault()
    FakeMemoryLayer.records.extend(
        [
            _chunk(
                memory_id="mem_sem",
                user_id="user_a",
                session_id="sess_1",
                memory_type=MemoryType.SEMANTIC,
            ),
            _chunk(
                memory_id="mem_proc",
                user_id="user_a",
                session_id="sess_1",
                memory_type=MemoryType.PROCEDURAL,
            ),
        ]
    )
    monkeypatch.setattr(cli_main, "MemoryLayer", FakeMemoryLayer)

    result = runner.invoke(
        cli_main.app,
        [
            "search",
            "what does user prefer",
            "--user-id",
            "user_a",
            "--memory-type",
            "semantic",
            "--memory-type",
            "procedural",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert len(payload["memories"]) == 2
    assert FakeMemoryLayer.recall_calls[0]["memory_types"] == [
        MemoryType.SEMANTIC,
        MemoryType.PROCEDURAL,
    ]


def test_delete_all_requires_yes_flag(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _reset_fake_memory_vault()
    monkeypatch.setattr(cli_main, "MemoryLayer", FakeMemoryLayer)

    result = runner.invoke(cli_main.app, ["delete", "--user-id", "user_a", "--all"])

    assert result.exit_code == 2


def test_stats_command_reports_counts(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _reset_fake_memory_vault()
    FakeMemoryLayer.records.extend(
        [
            _chunk(
                memory_id="mem_1",
                user_id="user_a",
                session_id="sess_1",
                memory_type=MemoryType.SEMANTIC,
            ),
            _chunk(
                memory_id="mem_2",
                user_id="user_a",
                session_id="sess_2",
                memory_type=MemoryType.EPISODIC,
                compressed=True,
            ),
            _chunk(
                memory_id="mem_3",
                user_id="user_a",
                session_id="sess_2",
                memory_type=MemoryType.PROCEDURAL,
            ),
        ]
    )
    monkeypatch.setattr(cli_main, "MemoryLayer", FakeMemoryLayer)

    result = runner.invoke(cli_main.app, ["stats", "--user-id", "user_a", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["total_memories"] == 3
    assert payload["sessions"] == 2
    assert payload["compressed_memories"] == 1
    assert payload["memory_by_type"]["semantic"] == 1
    assert payload["memory_by_type"]["episodic"] == 1
    assert payload["memory_by_type"]["procedural"] == 1


def test_compress_command_json_output(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _reset_fake_memory_vault()
    monkeypatch.setattr(cli_main, "MemoryLayer", FakeMemoryLayer)

    result = runner.invoke(
        cli_main.app,
        [
            "compress",
            "--user-id",
            "user_a",
            "--force",
            "--sessions-to-compress",
            "2",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["user_id"] == "user_a"
    assert payload["total_uncompressed_sessions"] == 3
    assert payload["sessions_compressed"] == 2
    assert payload["summaries_created"] == 2
    assert payload["memories_marked_compressed"] == 4
    assert payload["compressed_session_ids"] == ["sess_1", "sess_2"]
    assert FakeMemoryLayer.compress_calls == [
        {
            "force": True,
            "sessions_to_compress": 2,
        }
    ]


def test_list_command_plain_output_and_empty_state(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _reset_fake_memory_vault()
    monkeypatch.setattr(cli_main, "MemoryLayer", FakeMemoryLayer)

    empty_result = runner.invoke(cli_main.app, ["list", "--user-id", "user_a"])
    assert empty_result.exit_code == 0
    assert "No memories found." in empty_result.stdout

    FakeMemoryLayer.records.append(
        _chunk(
            memory_id="mem_plain",
            user_id="user_a",
            session_id="sess_1",
            memory_type=MemoryType.SEMANTIC,
        )
    )
    full_result = runner.invoke(cli_main.app, ["list", "--user-id", "user_a"])

    assert full_result.exit_code == 0
    assert "mem_plain" in full_result.stdout
    assert "Total: 1" in full_result.stdout


def test_search_command_plain_output_and_prompt(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _reset_fake_memory_vault()
    monkeypatch.setattr(cli_main, "MemoryLayer", FakeMemoryLayer)

    empty_result = runner.invoke(cli_main.app, ["search", "q", "--user-id", "user_a"])
    assert empty_result.exit_code == 0
    assert "No memories matched this query." in empty_result.stdout

    FakeMemoryLayer.records.append(
        _chunk(
            memory_id="mem_search",
            user_id="user_a",
            session_id="sess_1",
            memory_type=MemoryType.SEMANTIC,
        )
    )
    full_result = runner.invoke(
        cli_main.app,
        ["search", "q", "--user-id", "user_a", "--show-prompt"],
    )

    assert full_result.exit_code == 0
    assert "mem_search" in full_result.stdout
    assert "Tokens used:" in full_result.stdout
    assert "<memory>" in full_result.stdout


def test_delete_command_variants(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _reset_fake_memory_vault()
    monkeypatch.setattr(cli_main, "MemoryLayer", FakeMemoryLayer)

    invalid_combo = runner.invoke(
        cli_main.app,
        ["delete", "--user-id", "user_a", "--all", "--memory-id", "mem_1", "--yes"],
    )
    assert invalid_combo.exit_code == 2

    missing_memory_id = runner.invoke(cli_main.app, ["delete", "--user-id", "user_a"])
    assert missing_memory_id.exit_code == 2

    FakeMemoryLayer.records.append(
        _chunk(
            memory_id="mem_delete",
            user_id="user_a",
            session_id="sess_1",
            memory_type=MemoryType.SEMANTIC,
        )
    )

    deleted = runner.invoke(
        cli_main.app,
        ["delete", "--user-id", "user_a", "--memory-id", "mem_delete"],
    )
    assert deleted.exit_code == 0
    assert "Deleted memory mem_delete." in deleted.stdout

    not_found = runner.invoke(
        cli_main.app,
        ["delete", "--user-id", "user_a", "--memory-id", "mem_missing"],
    )
    assert not_found.exit_code == 0
    assert "was not found" in not_found.stdout

    FakeMemoryLayer.records.extend(
        [
            _chunk(
                memory_id="mem_all_1",
                user_id="user_a",
                session_id="sess_2",
                memory_type=MemoryType.EPISODIC,
            ),
            _chunk(
                memory_id="mem_all_2",
                user_id="user_a",
                session_id="sess_3",
                memory_type=MemoryType.PROCEDURAL,
            ),
        ]
    )
    deleted_all = runner.invoke(
        cli_main.app,
        ["delete", "--user-id", "user_a", "--all", "--yes"],
    )
    assert deleted_all.exit_code == 0
    assert "Deleted 2 memory records for user user_a." in deleted_all.stdout


def test_stats_and_compress_plain_output(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _reset_fake_memory_vault()
    monkeypatch.setattr(cli_main, "MemoryLayer", FakeMemoryLayer)

    stats_empty = runner.invoke(cli_main.app, ["stats", "--user-id", "user_a"])
    assert stats_empty.exit_code == 0
    assert "Total memories" in stats_empty.stdout

    compress_plain = runner.invoke(cli_main.app, ["compress", "--user-id", "user_a"])
    assert compress_plain.exit_code == 0
    assert "Compression result for user_a" in compress_plain.stdout


def test_mcp_tools_plain_output() -> None:
    result = runner.invoke(cli_main.app, ["mcp", "tools"])

    assert result.exit_code == 0
    assert "MCP tools" in result.stdout
    for tool_name in cli_main.TOOL_NAMES:
        assert tool_name in result.stdout


def test_mcp_start_success_and_failure(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[tuple[str, int]] = []

    def _ok(*, host: str, port: int) -> None:
        calls.append((host, port))

    monkeypatch.setattr(cli_main, "run_mcp_server", _ok)
    success = runner.invoke(cli_main.app, ["mcp", "start", "--host", "0.0.0.0", "--port", "8123"])

    assert success.exit_code == 0
    assert calls == [("0.0.0.0", 8123)]

    def _fail(*, host: str, port: int) -> None:
        del host, port
        raise RuntimeError("mcp failed")

    monkeypatch.setattr(cli_main, "run_mcp_server", _fail)
    failed = runner.invoke(cli_main.app, ["mcp", "start"])

    assert failed.exit_code == 1
    assert "mcp failed" in failed.stdout


def test_preview_truncates_and_main_invokes_app(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    assert cli_main._preview("a b c", limit=20) == "a b c"
    assert cli_main._preview("x" * 30, limit=10) == "xxxxxxx..."

    called = {"value": False}

    def _fake_app() -> None:
        called["value"] = True

    monkeypatch.setattr(cli_main, "app", _fake_app)
    cli_main.main()

    assert called["value"] is True
