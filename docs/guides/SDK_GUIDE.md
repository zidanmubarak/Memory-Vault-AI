# SDK Guide — Memory Vault AI

> **Audience:** Developers integrating Memory Vault AI as a Python library.
> For REST API usage, see [`docs/api/API_SPEC.md`](../api/API_SPEC.md).
> For AI agent integration (Claude Code, Cursor), see [`docs/guides/MCP_INTEGRATION.md`](MCP_INTEGRATION.md).

---

## Installation

```bash
# Standard install
pip install memory-vault

# With Qdrant support (production)
pip install "memory-vault[qdrant]"

# With all optional dependencies
pip install "memory-vault[all]"

# Development install from source
git clone https://github.com/zidanmubarak/Memory-Vault-AI
cd memory-vault-ai
pip install -e ".[dev]"
```

**Requirements:** Python 3.11+

---

## Quick Start

```python
import asyncio
from memory_vault import MemoryLayer

async def main():
    # Embedded mode — no server needed, data stored locally
    memory = MemoryLayer(user_id="alice")

    # Save context from a conversation turn
    await memory.save("I'm building a SaaS product using FastAPI and PostgreSQL.")
    await memory.save("I prefer concise answers with code examples, not long explanations.")

    # In a new session — recall relevant context
    context = await memory.recall("What stack is the user using?")
    print(context.prompt_block)
    # <memory>
    # [SEMANTIC] User is building a SaaS product with FastAPI and PostgreSQL.
    # [PROCEDURAL] Communication style: concise, code-first.
    # </memory>

asyncio.run(main())
```

---

## The `MemoryLayer` Class

### Constructor

```python
from memory_vault import MemoryLayer, MemoryConfig

memory = MemoryLayer(
    user_id="alice",              # Required. All memory is scoped to this ID.
    session_id="sess_abc123",     # Optional. Auto-generated if not provided.
    config=MemoryConfig(          # Optional. Uses defaults if omitted.
        token_budget=2000,
        top_k=5,
        embedding_model="all-MiniLM-L6-v2",
        storage_backend="chroma",
        chroma_path="./data/chroma",
        sqlite_path="./data/memory.db",
        compression_threshold=10,
    )
)
```

### `save()` — Store a memory

```python
chunks = await memory.save(
    text="I'm migrating from MySQL to PostgreSQL this quarter.",
    memory_type_hint=None,    # Optional. Auto-classified if omitted.
    session_id=None,          # Optional. Uses instance session_id if omitted.
)
# Returns: list[MemoryChunk] — the chunks that were saved (importance < 0.3 are discarded)
```

**Memory type hints:** `"episodic"`, `"semantic"`, `"working"`, `"procedural"`.
Use hints only when you know the type with certainty. Auto-classification is usually accurate.

**Batch save:**
```python
turns = [
    "User asked about database indexing strategies.",
    "Recommended B-tree indexes for range queries.",
    "User prefers to avoid ORM abstractions for complex queries.",
]
for turn in turns:
    await memory.save(turn)
```

### `recall()` — Retrieve relevant context

```python
result = await memory.recall(
    query="What database is the user using?",
    top_k=5,                  # Max memories to return
    token_budget=2000,        # Max tokens across all returned memories
    memory_types=None,        # None = all types. Or: ["semantic", "procedural"]
    include_compressed=False, # Include archived (compressed) memories
)

# result is a RecallResult object
print(result.memories)       # list[MemoryChunk]
print(result.total_tokens)   # int — tokens used
print(result.budget_used)    # float — 0.0 to 1.0
print(result.prompt_block)   # str — ready to inject into your LLM prompt
```

**Using `prompt_block` with an LLM:**
```python
result = await memory.recall("Tell me about the user")

system_prompt = f"""You are a helpful assistant.

{result.prompt_block}

Use the memory above to personalize your responses."""

# Pass system_prompt to your LLM call
```

### `list()` — Browse stored memories

```python
memories = await memory.list(
    memory_type="semantic",   # Filter by type. None = all.
    page=1,
    page_size=20,
    include_compressed=False,
)
# Returns: PaginatedResult[MemoryChunk]
print(memories.items)    # list[MemoryChunk]
print(memories.total)    # int — total count across all pages
```

### `forget()` — Delete a memory

```python
# Delete a specific memory
await memory.forget(memory_id="mem_abc123")

# Delete all memories for this user (GDPR, account deletion)
await memory.forget_all(confirm=True)
```

### `compress()` — Manually trigger compression

```python
# Compress old episodic memories into semantic summaries
job = await memory.compress(force=False, sessions_to_compress=None)
print(job.sessions_compressed)  # int
print(job.summaries_created)    # int — semantic summaries created
```

---

## Configuration Reference

All options can be set via `MemoryConfig` or environment variables.

| Config field | Env variable | Default | Description |
|---|---|---|---|
| `token_budget` | `ML_DEFAULT_TOKEN_BUDGET` | `2000` | Max tokens injected per recall |
| `top_k` | `ML_DEFAULT_TOP_K` | `5` | Max memories returned |
| `embedding_model` | `ML_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `storage_backend` | `ML_STORAGE_BACKEND` | `chroma` | `chroma` or `qdrant` |
| `chroma_path` | `ML_CHROMA_PATH` | `./data/chroma` | ChromaDB persistence path |
| `sqlite_path` | `ML_SQLITE_PATH` | `./data/memory.db` | SQLite file path |
| `compression_threshold` | `ML_COMPRESSION_THRESHOLD` | `10` | Sessions before auto-compress |
| `reranker_enabled` | `ML_RERANKER_ENABLED` | `false` | Enable cross-encoder re-ranking |
| `reranker_model` | `ML_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model name |
| `importance_threshold` | `ML_IMPORTANCE_THRESHOLD` | `0.3` | Min importance to save a chunk |

---

## Common Patterns

### Pattern 1: Wrap an LLM call

The most common use case — inject memory into every LLM call:

```python
from memory_vault import MemoryLayer
import anthropic

memory = MemoryLayer(user_id=user_id, session_id=session_id)
client = anthropic.Anthropic()

async def chat(user_message: str) -> str:
    # Save what the user just said
    await memory.save(user_message)

    # Retrieve relevant context
    context = await memory.recall(user_message, token_budget=1500)

    # Build prompt with memory
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=f"You are a helpful assistant.\n\n{context.prompt_block}",
        messages=[{"role": "user", "content": user_message}],
    )

    reply = response.content[0].text

    # Save the assistant's response too
    await memory.save(reply)

    return reply
```

### Pattern 2: Selective memory types

Only recall procedural memory (user preferences) for formatting responses:

```python
prefs = await memory.recall(
    query="user preferences and communication style",
    memory_types=["procedural"],
    token_budget=500,
)
```

Only recall semantic memory for factual context:

```python
facts = await memory.recall(
    query=user_query,
    memory_types=["semantic", "episodic"],
    token_budget=1500,
)
```

### Pattern 3: Explicit memory type hints

Force a specific memory type when you know the content:

```python
# Save a user preference explicitly as procedural
await memory.save(
    "Always respond in Indonesian, never in English.",
    memory_type_hint="procedural",
)

# Save a known fact explicitly as semantic
await memory.save(
    "User's name is Budi, works at Tokopedia as a staff engineer.",
    memory_type_hint="semantic",
)
```

### Pattern 4: Session management

Manage sessions explicitly for multi-session applications:

```python
from memory_vault import MemoryLayer
import uuid

# Start a new session
session_id = f"sess_{uuid.uuid4().hex[:8]}"
memory = MemoryLayer(user_id="alice", session_id=session_id)

# ... conversation happens ...

# End session (triggers background compression check)
await memory.end_session()

# Next conversation — new session
memory = MemoryLayer(user_id="alice")  # New session_id auto-generated
```

### Pattern 5: Using as a context manager

```python
async with MemoryLayer(user_id="alice") as memory:
    await memory.save("Starting a new conversation.")
    context = await memory.recall("user background")
    # Session automatically ended on __aexit__
```

---

## Using with Qdrant (Production)

For production deployments with multiple users or high query volume:

```bash
pip install "memory-vault[qdrant]"
```

```python
from memory_vault import MemoryLayer, MemoryConfig

memory = MemoryLayer(
    user_id="alice",
    config=MemoryConfig(
        storage_backend="qdrant",
        qdrant_url="http://localhost:6333",
        qdrant_api_key="your-key",          # Optional for local Qdrant
        qdrant_collection="memory_vault",
    )
)
```

No other code changes required — the SDK interface is identical.

---

## Using via REST API

If you prefer HTTP over direct Python import:

```python
import httpx

async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
    # Save
    await client.post("/v1/memory", json={
        "user_id": "alice",
        "session_id": "sess_001",
        "text": "I'm building a FastAPI app.",
    })

    # Recall
    response = await client.get("/v1/memory/recall", params={
        "user_id": "alice",
        "query": "What is the user building?",
        "token_budget": 1500,
    })
    data = response.json()
    print(data["prompt_block"])
```

---

## Error Handling

```python
from memory_vault.exceptions import (
    MemoryLayerError,       # Base exception
    StorageError,           # ChromaDB or SQLite failure
    EmbeddingError,         # Embedding model failure
    BudgetExceededError,    # token_budget too small for any memory
    UserNotFoundError,      # No memories exist for user_id
)

try:
    context = await memory.recall("user preferences")
except BudgetExceededError as e:
    # token_budget is too small to include even the smallest memory
    # Increase token_budget or filter to a specific memory_type
    print(f"Budget too small: {e.minimum_required} tokens needed")
except StorageError as e:
    # Storage backend is unavailable
    print(f"Storage error: {e}")
```

---

## Performance Tips

- **Batch saves:** Call `save()` once per conversation turn, not once per sentence.
- **Right-size `top_k`:** `top_k=3` is enough for most use cases. Higher values add latency.
- **Set `token_budget` to ~20% of your model's context window** to leave room for conversation.
- **Disable reranker** (`reranker_enabled=False`) unless retrieval quality is critical — it adds ~200ms.
- **Local embedding model** (`all-MiniLM-L6-v2`) is fast (~5ms/chunk on CPU). No API calls needed.
