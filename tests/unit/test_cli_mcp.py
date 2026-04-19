from __future__ import annotations

import importlib
import json
from typing import Any

from typer.testing import CliRunner

cli_main = importlib.import_module("memory_layer.cli.main")
runner = CliRunner()


def test_mcp_tools_command_json_output() -> None:
    result = runner.invoke(cli_main.app, ["mcp", "tools", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["tools"] == [
        "memory_save",
        "memory_recall",
        "memory_list",
        "memory_forget",
    ]


def test_mcp_start_command_passes_host_and_port(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, Any] = {}

    def _fake_run_mcp_server(*, host: str, port: int) -> None:
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr(cli_main, "run_mcp_server", _fake_run_mcp_server)

    result = runner.invoke(
        cli_main.app,
        ["mcp", "start", "--host", "0.0.0.0", "--port", "9001"],
    )

    assert result.exit_code == 0
    assert captured == {"host": "0.0.0.0", "port": 9001}


def test_mcp_start_command_reports_runtime_error(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def _failing_run_mcp_server(*, host: str, port: int) -> None:
        del host, port
        raise RuntimeError("uvicorn missing")

    monkeypatch.setattr(cli_main, "run_mcp_server", _failing_run_mcp_server)

    result = runner.invoke(cli_main.app, ["mcp", "start"])

    assert result.exit_code == 1
    assert "uvicorn missing" in result.stdout
