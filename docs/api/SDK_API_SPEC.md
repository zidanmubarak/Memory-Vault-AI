# SDK API Specification - Memory Vault AI

> Versioned public Python SDK contract.
> This specification defines semver-protected API surface for `MemoryLayer`.

---

## Stability Policy

- The public SDK contract is semver-governed.
- Breaking changes to this contract require a major version bump.
- Backward-compatible additions (new optional keyword args, new methods, new exports) are minor releases.
- Patch releases must not change public signatures or response/result model semantics.
- Deprecated APIs must emit deprecation guidance and remain available for at least one minor release before removal.

---

## Public Imports

Supported imports:

```python
from memory_vault import MemoryLayer
from memory_vault.sdk import MemoryLayer, SDK_PUBLIC_METHODS
```

`SDK_PUBLIC_METHODS` is the canonical list of semver-protected async methods.

---

## Constructor Contract

```python
MemoryLayer(
    *,
    user_id: str,
    session_id: str | None = None,
    config: MemoryConfig | None = None,
    settings: Settings | None = None,
    storage: StorageBackend | None = None,
    ingestion_engine: IngestionPipeline | None = None,
    retrieval_engine: RetrievalPipeline | None = None,
)
```

Contract notes:
- `user_id` is required and keyword-only.
- Constructor parameters are keyword-only and order-preserving.
- Constructor may raise `ValueError` (`user_id`) and `ConfigurationError` (unsupported backend combinations).

---

## Stable Async Method Surface

```python
async def initialize(self) -> None
async def close(self) -> None

async def save(
    self,
    text: str,
    *,
    memory_type_hint: MemoryType | None = None,
    session_id: str | None = None,
) -> list[MemoryChunk]

async def recall(
    self,
    query: str,
    *,
    top_k: int | None = None,
    token_budget: int | None = None,
    memory_types: list[MemoryType] | None = None,
    include_compressed: bool = False,
    reranker_enabled: bool | None = None,
) -> RecallResult

async def list(
    self,
    *,
    memory_type: MemoryType | None = None,
    page: int = 1,
    page_size: int = 20,
    include_compressed: bool = False,
) -> PaginatedResult[MemoryChunk]

async def upsert_procedural_memory(
    self,
    *,
    key: str,
    value: str,
    confidence: float = 1.0,
    source_chunk_id: str | None = None,
) -> ProceduralMemoryRecord

async def list_procedural_memory(self) -> Sequence[ProceduralMemoryRecord]
async def delete_procedural_memory(self, *, key: str) -> bool

async def compress(
    self,
    *,
    force: bool = False,
    sessions_to_compress: int | None = None,
) -> CompressionResult

async def forget(self, *, memory_id: str) -> bool
async def forget_all(self, *, confirm: bool = False) -> int
async def end_session(self) -> SessionStatsRecord | None
```

---

## Non-Contract Internals

The following are explicitly not semver-stable:
- private attributes and methods (`_storage`, `_build_storage`, `_new_embedder`, etc.)
- internal engine classes and wiring details
- test-only stubs/fakes

---

## Contract Enforcement

This SDK contract is enforced by unit tests in:
- `tests/unit/test_sdk_api_contract.py`

These tests assert:
- exported SDK symbol surface
- constructor parameter stability
- async method signature stability for `SDK_PUBLIC_METHODS`
