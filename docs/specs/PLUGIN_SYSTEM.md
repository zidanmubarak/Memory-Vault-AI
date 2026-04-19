# Plugin System Specification

> This spec defines the custom memory type plugin system used by ingestion.

---

## Goal

Allow integrators to define **custom memory type classifiers** without changing core library code.

Custom memory types are represented as plugin names and mapped onto existing core `MemoryType` values for storage/retrieval compatibility.

---

## Plugin Interface

A plugin must provide:

- `name: str` — unique identifier for custom type (for example: `project_profile`)
- `base_memory_type: MemoryType` — core storage type (`semantic`, `episodic`, etc.)
- `priority: int` — higher value evaluated first (default: `100`)
- `matches(chunk_text: str) -> bool` — classifier predicate
- `metadata(chunk_text: str) -> Mapping[str, object] | None` — optional metadata payload

---

## Registration

Plugins are registered in a `MemoryTypePluginRegistry`.

The default process-wide registry is available through:

- `register_memory_type_plugin(plugin)`
- `unregister_memory_type_plugin(name)`
- `clear_memory_type_plugins()`
- `get_default_plugin_registry()`

Registry behavior:

- names are normalized to lowercase
- duplicate names are rejected
- plugins are ordered by `priority` descending, then name ascending
- plugin exceptions during matching are ignored so ingestion remains available

---

## Ingestion Resolution Order

When ingesting a chunk with no explicit `memory_type_hint`:

1. evaluate registered plugins in registry order
2. if a plugin matches:
   - persist chunk using `plugin.base_memory_type`
   - attach `metadata["custom_memory_type"] = plugin.name`
   - merge plugin metadata payload (excluding reserved `custom_memory_type` key)
3. if no plugin matches: use built-in heuristic classification rules

When `memory_type_hint` is explicitly provided, plugin classification is bypassed.

---

## Prompt Formatting

Prompt rendering should surface custom type labels when present:

- default label: `[SEMANTIC]`
- custom label: `[SEMANTIC:PROJECT_PROFILE]`

---

## Compatibility

- Storage and API contracts remain compatible because persisted `memory_type` still uses `MemoryType`.
- Custom type identity is carried in chunk metadata via `custom_memory_type`.
- Existing SDK public method signatures remain unchanged.
