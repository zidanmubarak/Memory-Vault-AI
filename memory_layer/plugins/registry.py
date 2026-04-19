from __future__ import annotations

from threading import RLock

from memory_layer.models import MemoryType
from memory_layer.plugins.base import MemoryTypePlugin


class MemoryTypePluginRegistry:
    """Thread-safe registry for custom memory type plugins."""

    def __init__(self) -> None:
        self._plugins: dict[str, MemoryTypePlugin] = {}
        self._lock = RLock()

    def register(self, plugin: MemoryTypePlugin) -> None:
        """Register a plugin by unique normalized plugin name."""
        plugin_name = self._normalize_name(plugin.name)
        if not isinstance(plugin.base_memory_type, MemoryType):
            raise ValueError("plugin.base_memory_type must be a MemoryType")

        with self._lock:
            if plugin_name in self._plugins:
                raise ValueError(f"plugin '{plugin_name}' is already registered")
            self._plugins[plugin_name] = plugin

    def unregister(self, name: str) -> bool:
        """Unregister a plugin by name; returns True when removed."""
        plugin_name = self._normalize_name(name)
        with self._lock:
            return self._plugins.pop(plugin_name, None) is not None

    def get(self, name: str) -> MemoryTypePlugin | None:
        """Return one plugin by normalized name when registered."""
        plugin_name = self._normalize_name(name)
        with self._lock:
            return self._plugins.get(plugin_name)

    def clear(self) -> None:
        """Remove all plugins from the registry."""
        with self._lock:
            self._plugins.clear()

    def list_plugins(self) -> list[MemoryTypePlugin]:
        """Return registered plugins sorted by priority then name."""
        with self._lock:
            plugins = list(self._plugins.values())
        return sorted(
            plugins,
            key=lambda plugin: (-plugin.priority, plugin.name.lower()),
        )

    def match(self, chunk_text: str) -> MemoryTypePlugin | None:
        """Return first matching plugin for text, or None when no plugin matches."""
        for plugin in self.list_plugins():
            try:
                if plugin.matches(chunk_text):
                    return plugin
            except Exception:
                continue
        return None

    @staticmethod
    def _normalize_name(name: str) -> str:
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("plugin name is required")
        return normalized


_DEFAULT_PLUGIN_REGISTRY = MemoryTypePluginRegistry()


def get_default_plugin_registry() -> MemoryTypePluginRegistry:
    """Return process-wide default plugin registry instance."""
    return _DEFAULT_PLUGIN_REGISTRY


def register_memory_type_plugin(plugin: MemoryTypePlugin) -> None:
    """Register a plugin in the default registry."""
    _DEFAULT_PLUGIN_REGISTRY.register(plugin)


def unregister_memory_type_plugin(name: str) -> bool:
    """Remove a plugin from the default registry by name."""
    return _DEFAULT_PLUGIN_REGISTRY.unregister(name)


def clear_memory_type_plugins() -> None:
    """Remove all plugins from the default registry."""
    _DEFAULT_PLUGIN_REGISTRY.clear()


__all__ = [
    "MemoryTypePluginRegistry",
    "clear_memory_type_plugins",
    "get_default_plugin_registry",
    "register_memory_type_plugin",
    "unregister_memory_type_plugin",
]
