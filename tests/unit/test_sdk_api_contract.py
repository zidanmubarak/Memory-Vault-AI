from __future__ import annotations

import inspect
from collections.abc import Callable

import memory_vault
from memory_vault import MemoryLayer
from memory_vault.sdk import SDK_PUBLIC_METHODS

_EMPTY = inspect.Signature.empty


def _signature_contract(
    func: Callable[..., object],
) -> list[tuple[str, inspect._ParameterKind, object]]:
    signature = inspect.signature(func)
    return [
        (parameter.name, parameter.kind, parameter.default)
        for parameter in signature.parameters.values()
    ]


def test_sdk_top_level_exports_include_stable_symbols() -> None:
    exported = set(memory_vault.__all__)
    expected = {
        "MemoryLayer",
        "MemoryConfig",
        "MemoryChunk",
        "MemoryType",
        "PaginatedResult",
        "RecallResult",
        "SaveResult",
        "Settings",
        "get_settings",
    }
    assert expected.issubset(exported)


def test_sdk_module_exports_memory_vault_and_contract_list() -> None:
    from memory_vault import sdk

    assert sdk.__all__ == ["SDK_PUBLIC_METHODS", "MemoryLayer"]


def test_memory_vault_constructor_signature_is_stable() -> None:
    contract = _signature_contract(MemoryLayer.__init__)

    assert contract == [
        ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, _EMPTY),
        ("user_id", inspect.Parameter.KEYWORD_ONLY, _EMPTY),
        ("session_id", inspect.Parameter.KEYWORD_ONLY, None),
        ("config", inspect.Parameter.KEYWORD_ONLY, None),
        ("settings", inspect.Parameter.KEYWORD_ONLY, None),
        ("storage", inspect.Parameter.KEYWORD_ONLY, None),
        ("ingestion_engine", inspect.Parameter.KEYWORD_ONLY, None),
        ("retrieval_engine", inspect.Parameter.KEYWORD_ONLY, None),
    ]


def test_sdk_public_methods_are_async() -> None:
    for method_name in SDK_PUBLIC_METHODS:
        method = getattr(MemoryLayer, method_name)
        assert inspect.iscoroutinefunction(method), method_name


def test_sdk_public_method_signatures_are_stable() -> None:
    expected: dict[str, list[tuple[str, inspect._ParameterKind, object]]] = {
        "initialize": [
            ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, _EMPTY),
        ],
        "close": [
            ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, _EMPTY),
        ],
        "save": [
            ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, _EMPTY),
            ("text", inspect.Parameter.POSITIONAL_OR_KEYWORD, _EMPTY),
            ("memory_type_hint", inspect.Parameter.KEYWORD_ONLY, None),
            ("session_id", inspect.Parameter.KEYWORD_ONLY, None),
        ],
        "recall": [
            ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, _EMPTY),
            ("query", inspect.Parameter.POSITIONAL_OR_KEYWORD, _EMPTY),
            ("top_k", inspect.Parameter.KEYWORD_ONLY, None),
            ("token_budget", inspect.Parameter.KEYWORD_ONLY, None),
            ("memory_types", inspect.Parameter.KEYWORD_ONLY, None),
            ("include_compressed", inspect.Parameter.KEYWORD_ONLY, False),
            ("reranker_enabled", inspect.Parameter.KEYWORD_ONLY, None),
        ],
        "list": [
            ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, _EMPTY),
            ("memory_type", inspect.Parameter.KEYWORD_ONLY, None),
            ("page", inspect.Parameter.KEYWORD_ONLY, 1),
            ("page_size", inspect.Parameter.KEYWORD_ONLY, 20),
            ("include_compressed", inspect.Parameter.KEYWORD_ONLY, False),
        ],
        "upsert_procedural_memory": [
            ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, _EMPTY),
            ("key", inspect.Parameter.KEYWORD_ONLY, _EMPTY),
            ("value", inspect.Parameter.KEYWORD_ONLY, _EMPTY),
            ("confidence", inspect.Parameter.KEYWORD_ONLY, 1.0),
            ("source_chunk_id", inspect.Parameter.KEYWORD_ONLY, None),
        ],
        "list_procedural_memory": [
            ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, _EMPTY),
        ],
        "delete_procedural_memory": [
            ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, _EMPTY),
            ("key", inspect.Parameter.KEYWORD_ONLY, _EMPTY),
        ],
        "compress": [
            ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, _EMPTY),
            ("force", inspect.Parameter.KEYWORD_ONLY, False),
            ("sessions_to_compress", inspect.Parameter.KEYWORD_ONLY, None),
        ],
        "forget": [
            ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, _EMPTY),
            ("memory_id", inspect.Parameter.KEYWORD_ONLY, _EMPTY),
        ],
        "forget_all": [
            ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, _EMPTY),
            ("confirm", inspect.Parameter.KEYWORD_ONLY, False),
        ],
        "end_session": [
            ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, _EMPTY),
        ],
    }

    assert tuple(expected) == SDK_PUBLIC_METHODS
    for method_name, signature in expected.items():
        assert _signature_contract(getattr(MemoryLayer, method_name)) == signature
