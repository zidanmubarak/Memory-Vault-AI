from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, ClassVar

from fastapi.testclient import TestClient

from memory_layer.config import Settings
from memory_layer.mcp.server import create_mcp_app
from memory_layer.models import MemoryChunk, MemoryType, PaginatedResult, RecallResult


class FakeMemoryLayer:
    records: ClassVar[list[MemoryChunk]] = []

    def __init__(
        self,
        *,
        user_id: str,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.user_id = user_id
        self.session_id = session_id or "sess_default"

    async def close(self) -> None:
        return None

    async def save(
        self,
        text: str,
        *,
        memory_type_hint: MemoryType | None = None,
        session_id: str | None = None,
    ) -> list[MemoryChunk]:
        now = datetime(2026, 4, 19, 10, 0, tzinfo=UTC)
        chunk = MemoryChunk(
            id=f"mem_{len(FakeMemoryLayer.records) + 1}",
            user_id=self.user_id,
            session_id=session_id or self.session_id,
            memory_type=memory_type_hint or MemoryType.EPISODIC,
            content=text,
            importance=0.8,
            token_count=6,
            embedding=[0.1, 0.2],
            created_at=now,
            updated_at=now,
        )
        FakeMemoryLayer.records.append(chunk)
        return [chunk]

    async def recall(
        self,
        query: str,
        *,
        top_k: int = 5,
        token_budget: int = 2000,
        memory_types: list[MemoryType] | None = None,
        include_compressed: bool = False,
        reranker_enabled: bool | None = None,
    ) -> RecallResult:
        del query, reranker_enabled
        selected = [chunk for chunk in FakeMemoryLayer.records if chunk.user_id == self.user_id]
        if memory_types:
            selected = [chunk for chunk in selected if chunk.memory_type in memory_types]
        if not include_compressed:
            selected = [chunk for chunk in selected if not chunk.compressed]

        selected = selected[:top_k]
        total_tokens = sum(chunk.token_count for chunk in selected)
        budget_used = 0.0 if token_budget == 0 else total_tokens / token_budget

        return RecallResult(
            memories=selected,
            total_tokens=total_tokens,
            budget_used=budget_used,
            prompt_block="<memory>\n</memory>",
        )

    async def list(
        self,
        *,
        memory_type: MemoryType | None = None,
        page: int = 1,
        page_size: int = 20,
        include_compressed: bool = False,
    ) -> PaginatedResult[MemoryChunk]:
        selected = [chunk for chunk in FakeMemoryLayer.records if chunk.user_id == self.user_id]
        if memory_type is not None:
            selected = [chunk for chunk in selected if chunk.memory_type is memory_type]
        if not include_compressed:
            selected = [chunk for chunk in selected if not chunk.compressed]

        start = (page - 1) * page_size
        end = start + page_size
        return PaginatedResult[MemoryChunk](
            items=selected[start:end],
            total=len(selected),
            page=page,
            page_size=page_size,
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


def _reset_records() -> None:
    FakeMemoryLayer.records.clear()


def _rpc(method: str, *, rpc_id: int = 1, params: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "method": method,
    }
    if params is not None:
        payload["params"] = params
    return payload


def test_mcp_health_and_tools_list(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _reset_records()
    import memory_layer.mcp.server as mcp_server

    monkeypatch.setattr(mcp_server, "MemoryLayer", FakeMemoryLayer)

    app = create_mcp_app(settings=Settings(storage_backend="chroma", metadata_backend="sqlite"))

    with TestClient(app) as client:
        health_response = client.get("/mcp/v1/health")
        list_response = client.post("/mcp/v1", json=_rpc("tools/list"))

    assert health_response.status_code == 200
    assert health_response.json()["tools"] == [
        "memory_save",
        "memory_recall",
        "memory_list",
        "memory_forget",
    ]

    assert list_response.status_code == 200
    tools = list_response.json()["result"]["tools"]
    names = [tool["name"] for tool in tools]
    assert names == ["memory_save", "memory_recall", "memory_list", "memory_forget"]


def test_mcp_save_and_recall_tools(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _reset_records()
    import memory_layer.mcp.server as mcp_server

    monkeypatch.setattr(mcp_server, "MemoryLayer", FakeMemoryLayer)

    app = create_mcp_app(settings=Settings(storage_backend="chroma", metadata_backend="sqlite"))

    with TestClient(app) as client:
        save_response = client.post(
            "/mcp/v1",
            json=_rpc(
                "tools/call",
                params={
                    "name": "memory_save",
                    "arguments": {
                        "text": "Remember this",
                        "user_id": "user_a",
                        "session_id": "sess_1",
                        "memory_type": "semantic",
                    },
                },
            ),
        )
        recall_response = client.post(
            "/mcp/v1",
            json=_rpc(
                "tools/call",
                params={
                    "name": "memory_recall",
                    "arguments": {
                        "query": "What should you remember?",
                        "user_id": "user_a",
                        "top_k": 5,
                        "token_budget": 2000,
                    },
                },
            ),
        )

    assert save_response.status_code == 200
    save_result = save_response.json()["result"]["structuredContent"]
    assert save_result["saved"][0]["memory_type"] == "semantic"

    assert recall_response.status_code == 200
    recall_result = recall_response.json()["result"]["structuredContent"]
    assert len(recall_result["memories"]) == 1
    assert recall_result["memories"][0]["content"] == "Remember this"

    content_text = recall_response.json()["result"]["content"][0]["text"]
    structured_from_text = json.loads(content_text)
    assert structured_from_text["memories"][0]["id"] == recall_result["memories"][0]["id"]


def test_mcp_list_and_forget_tools(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _reset_records()
    import memory_layer.mcp.server as mcp_server

    monkeypatch.setattr(mcp_server, "MemoryLayer", FakeMemoryLayer)

    app = create_mcp_app(settings=Settings(storage_backend="chroma", metadata_backend="sqlite"))

    with TestClient(app) as client:
        client.post(
            "/mcp/v1",
            json=_rpc(
                "tools/call",
                params={
                    "name": "memory_save",
                    "arguments": {
                        "text": "One",
                        "user_id": "user_a",
                        "session_id": "sess_1",
                    },
                },
            ),
        )
        client.post(
            "/mcp/v1",
            json=_rpc(
                "tools/call",
                params={
                    "name": "memory_save",
                    "arguments": {
                        "text": "Two",
                        "user_id": "user_a",
                        "session_id": "sess_1",
                    },
                },
            ),
        )

        list_response = client.post(
            "/mcp/v1",
            json=_rpc(
                "tools/call",
                params={
                    "name": "memory_list",
                    "arguments": {
                        "user_id": "user_a",
                        "page": 1,
                        "page_size": 20,
                    },
                },
            ),
        )
        items = list_response.json()["result"]["structuredContent"]["items"]
        first_id = items[0]["id"]

        forget_one_response = client.post(
            "/mcp/v1",
            json=_rpc(
                "tools/call",
                params={
                    "name": "memory_forget",
                    "arguments": {
                        "user_id": "user_a",
                        "memory_id": first_id,
                    },
                },
            ),
        )

        forget_all_response = client.post(
            "/mcp/v1",
            json=_rpc(
                "tools/call",
                params={
                    "name": "memory_forget",
                    "arguments": {
                        "user_id": "user_a",
                        "confirm": True,
                    },
                },
            ),
        )

    assert list_response.status_code == 200
    assert list_response.json()["result"]["structuredContent"]["total"] == 2

    assert forget_one_response.status_code == 200
    forget_one = forget_one_response.json()["result"]["structuredContent"]
    assert forget_one["deleted"] is True
    assert forget_one["deleted_count"] == 1

    assert forget_all_response.status_code == 200
    forget_all = forget_all_response.json()["result"]["structuredContent"]
    assert forget_all["deleted"] is True
    assert forget_all["deleted_count"] == 1


def test_mcp_auth_enforced_when_api_key_configured(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _reset_records()
    import memory_layer.mcp.server as mcp_server

    monkeypatch.setattr(mcp_server, "MemoryLayer", FakeMemoryLayer)

    app = create_mcp_app(
        settings=Settings(
            storage_backend="chroma",
            metadata_backend="sqlite",
            api_key="secret",
        )
    )

    with TestClient(app) as client:
        unauthorized = client.post("/mcp/v1", json=_rpc("tools/list"))
        authorized = client.post(
            "/mcp/v1",
            json=_rpc("tools/list"),
            headers={"Authorization": "Bearer secret"},
        )
        health = client.get("/mcp/v1/health")

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    assert health.status_code == 200
