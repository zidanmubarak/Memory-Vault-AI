# Plugin System Guide — Custom Memory Types

Use plugins to add custom memory type classification behavior without modifying core ingestion code.

---

## Quick Example

```python
from memory_vault import MemoryLayer
from memory_vault.models import MemoryType
from memory_vault.plugins import MemoryTypePlugin, register_memory_type_plugin


class ProjectProfilePlugin(MemoryTypePlugin):
    name = "project_profile"
    base_memory_type = MemoryType.SEMANTIC
    priority = 200

    def matches(self, chunk_text: str) -> bool:
        lowered = chunk_text.lower()
        return "project profile" in lowered or "architecture summary" in lowered

    def metadata(self, chunk_text: str) -> dict[str, object]:
        del chunk_text
        return {"plugin_version": 1}


register_memory_type_plugin(ProjectProfilePlugin())

memory = MemoryLayer(user_id="alice")
```

When a chunk matches the plugin:

- it is stored as `base_memory_type` (`semantic` in this example)
- metadata includes `custom_memory_type="project_profile"`
- prompt labels include `[SEMANTIC:PROJECT_PROFILE]`

---

## API Reference

From `memory_vault.plugins`:

- `MemoryTypePlugin`
- `MemoryTypePluginRegistry`
- `register_memory_type_plugin(plugin)`
- `unregister_memory_type_plugin(name)`
- `clear_memory_type_plugins()`
- `get_default_plugin_registry()`

---

## Best Practices

- Keep `matches()` fast and deterministic.
- Prefer explicit keywords/patterns over heavy NLP in plugin code.
- Use higher `priority` only when a plugin must override other custom rules.
- Include compact metadata payloads to avoid prompt/storage bloat.

---

## Isolation In Tests

Use a local registry and pass it to `IngestionEngine(..., plugin_registry=...)` in unit tests.
This avoids leakage across tests from the process-wide default registry.
