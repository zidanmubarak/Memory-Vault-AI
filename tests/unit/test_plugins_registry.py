from __future__ import annotations

import pytest

from memory_layer.models import MemoryType
from memory_layer.plugins import (
    MemoryTypePlugin,
    MemoryTypePluginRegistry,
    clear_memory_type_plugins,
    get_default_plugin_registry,
    register_memory_type_plugin,
    unregister_memory_type_plugin,
)


class _KeywordPlugin(MemoryTypePlugin):
    def __init__(
        self,
        *,
        name: str,
        keyword: str,
        base_memory_type: MemoryType,
        priority: int = 100,
    ) -> None:
        self.name = name
        self.keyword = keyword
        self.base_memory_type = base_memory_type
        self.priority = priority

    def matches(self, chunk_text: str) -> bool:
        return self.keyword in chunk_text.lower()


def test_registry_register_get_list_unregister_and_clear() -> None:
    registry = MemoryTypePluginRegistry()
    plugin = _KeywordPlugin(
        name="project_profile",
        keyword="project profile",
        base_memory_type=MemoryType.SEMANTIC,
    )

    registry.register(plugin)

    assert registry.get("project_profile") is plugin
    assert registry.get("PROJECT_PROFILE") is plugin
    assert registry.list_plugins() == [plugin]

    assert registry.unregister("project_profile") is True
    assert registry.unregister("project_profile") is False

    registry.register(plugin)
    registry.clear()
    assert registry.list_plugins() == []


def test_registry_rejects_duplicate_plugin_name() -> None:
    registry = MemoryTypePluginRegistry()
    first = _KeywordPlugin(
        name="duplicate",
        keyword="first",
        base_memory_type=MemoryType.SEMANTIC,
    )
    second = _KeywordPlugin(
        name="DUPLICATE",
        keyword="second",
        base_memory_type=MemoryType.SEMANTIC,
    )

    registry.register(first)
    with pytest.raises(ValueError):
        registry.register(second)


def test_registry_prefers_higher_priority_match() -> None:
    registry = MemoryTypePluginRegistry()
    low = _KeywordPlugin(
        name="low",
        keyword="pipeline",
        base_memory_type=MemoryType.SEMANTIC,
        priority=10,
    )
    high = _KeywordPlugin(
        name="high",
        keyword="pipeline",
        base_memory_type=MemoryType.EPISODIC,
        priority=100,
    )

    registry.register(low)
    registry.register(high)

    matched = registry.match("this pipeline handles events")

    assert matched is high


def test_registry_ignores_plugin_exceptions_during_match() -> None:
    registry = MemoryTypePluginRegistry()

    class _FailingPlugin(MemoryTypePlugin):
        name = "failing"
        base_memory_type = MemoryType.SEMANTIC

        def matches(self, chunk_text: str) -> bool:
            del chunk_text
            raise RuntimeError("boom")

    fallback = _KeywordPlugin(
        name="fallback",
        keyword="keyword",
        base_memory_type=MemoryType.SEMANTIC,
    )

    registry.register(_FailingPlugin())
    registry.register(fallback)

    assert registry.match("keyword found") is fallback


def test_default_registry_helpers_register_and_unregister() -> None:
    clear_memory_type_plugins()
    plugin = _KeywordPlugin(
        name="default_registry_plugin",
        keyword="default",
        base_memory_type=MemoryType.SEMANTIC,
    )

    register_memory_type_plugin(plugin)
    default_registry = get_default_plugin_registry()

    assert default_registry.get("default_registry_plugin") is plugin
    assert unregister_memory_type_plugin("default_registry_plugin") is True
    assert default_registry.get("default_registry_plugin") is None
    clear_memory_type_plugins()
