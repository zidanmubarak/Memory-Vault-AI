# Architecture — Memory Vault AI

> **Audience:** AI coding agents, senior contributors, and anyone building integrations.
> For a quick overview, see the [documentation home](../index.md).
> For algorithm-level detail, see [Memory Logic](../specs/MEMORY_LOGIC.md).

---

## System Overview

Memory Vault AI is a **middleware system** that sits between a user-facing application and
any LLM. Its sole job is to manage what the model knows about the user across sessions.

```
┌───────────────────────────────────────────────────────────────────┐
│                    Client Application                             │
│              (chatbot, IDE plugin, voice assistant)               │
└──────────────────────────┬────────────────────────────────────────┘
                           │  HTTP  or  Python SDK
┌──────────────────────────▼────────────────────────────────────────┐
│                     Memory Vault AI                               │
│  ┌──────────────┐  ┌────────────┐  ┌──────────────────────────┐  │
│  │  API Layer   │  │  SDK       │  │  MCP Server              │  │
│  │  (FastAPI)   │  │  (Python)  │  │  (for AI agent tools)    │  │
│  └──────┬───────┘  └─────┬──────┘  └────────────┬─────────────┘  │
│         └────────────────┴──────────────────────┘                 │
│                          │                                        │
│  ┌───────────────────────▼───────────────────────────────────┐   │
│  │                   Core Engine                              │   │
│  │  Ingestion → Storage → Retrieval → Budget → Prompt Build  │   │
│  └───────────────────────────────────────────────────────────┘   │
│                          │                                        │
│  ┌───────────────────────▼───────────────────────────────────┐   │
│  │                  Storage Layer                             │   │
│  │         ChromaDB (vectors) + SQLite (metadata)            │   │
│  └───────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────────┐
│                     LLM Provider                                  │
│           Claude / GPT-4 / Ollama / any OpenAI-compatible         │
└───────────────────────────────────────────────────────────────────┘
```

---

## Component Reference

### 1. API Layer (`memory_vault/api/`)

FastAPI application exposing REST endpoints. Handles authentication, request validation,
and routes to core engine methods.

**Key files:**
- `main.py` — app factory and lifespan hooks
- `routes/memory.py` — memory CRUD endpoints
- `routes/session.py` — session management
- `middleware/auth.py` — API key validation
- `middleware/rate_limit.py` — per-user rate limiting

**Contracts:** See `docs/api/API_SPEC.md`. Do not change endpoint signatures without updating the spec.

---

### 2. Ingestion Engine (`memory_vault/ingestion/`)

Processes raw text into structured, embeddable memory chunks.

**Pipeline:**
```
raw_text
   → TextCleaner        (normalize whitespace, strip PII markers)
   → Chunker            (semantic chunking, not fixed-size)
   → EmbeddingModel     (sentence-transformers, async batched)
   → ImportanceScorer   (novelty + salience → float 0.0–1.0)
   → MemoryRouter       (decide: episodic / semantic / working / skip)
```

**Key classes:**
- `IngestionEngine` — orchestrator (public interface)
- `SemanticChunker` — splits text at natural boundaries
- `ImportanceScorer` — scores chunks using cosine similarity to existing memory
- `MemoryRouter` — classifies memory type based on content + score

**Decision: why semantic chunking?**
See `docs/adr/ADR-002-chunking-strategy.md`.

---

### 3. Storage Layer (`memory_vault/storage/`)

Abstraction over all persistence backends. **Feature code must never call ChromaDB or
SQLite directly** — always use `StorageLayer`.

```
StorageLayer (abstract base)
├── ChromaAdapter          — vector storage, similarity search
├── SQLiteAdapter          — metadata, session tracking, procedural memory
└── CompositeStorage       — coordinates both, ensures consistency
```

**Memory type → backend mapping:**

| Memory Type | Vectors (ChromaDB) | Metadata (SQLite) |
|---|---|---|
| Episodic | ✓ full content | session_id, timestamp, importance |
| Semantic | ✓ full content | entity type, confidence, source session |
| Working | in-memory only | session_id, ttl |
| Procedural | ✗ | key-value store in SQLite |

**Schema:** See `docs/specs/DATABASE_SCHEMA.md`.

---

### 4. Retrieval Engine (`memory_vault/retrieval/`)

Finds the most relevant memories for a given user query.

**Pipeline:**
```
query_text
   → QueryEmbedder        (same model as ingestion)
   → ANNSearch            (ChromaDB approximate nearest neighbor)
   → CandidateFilter      (remove stale, low-importance, or irrelevant)
   → CrossEncoderReranker (optional: more accurate relevance scoring)
   → MemoryCompressor     (summarize long chunks to save tokens)
   → RecallResult         (list of MemoryChunk, total_tokens)
```

**Key tunable parameters** (set via config or env):
- `top_k_candidates` — how many ANN results to fetch (default: 20)
- `top_k_return` — how many to return after re-ranking (default: 5)
- `reranker_enabled` — enable cross-encoder re-ranking (default: false, adds latency)
- `staleness_days` — deprioritize memories older than N days

---

### 5. Context Budget Manager (`memory_vault/budget/`)

Enforces token limits so retrieved memories never overflow the LLM's context window.

**Algorithm:**
1. Count tokens in all retrieved memories using `tiktoken` (cl100k_base by default)
2. Sort memories by relevance score (descending)
3. Greedily include memories until `token_budget` is exhausted
4. Return included memories + token usage stats

**The budget is set per-call**, not globally, so callers can tune per-model.

---

### 6. Memory Compression Engine (`memory_vault/compression/`)

Background job that runs when episodic memory for a user exceeds `compression_threshold`
sessions. Summarizes old episodes to free storage and keep retrieval quality high.

**Strategy:**
- Group episodic memories by session
- Sessions older than threshold: summarize with LLM into a single `semantic` memory
- Original episodic memories are archived (not deleted) and marked `compressed=True`
- Compression runs as a background `asyncio` task, never blocking request handling

**LLM used for compression:** Configured via `ML_COMPRESSION_MODEL` (default: cheapest available).

---

### 7. Prompt Builder (`memory_vault/prompt/`)

Assembles the final context block to inject into the LLM prompt.

**Output format:**
```
<memory>
[Semantic] Alice is a backend engineer at a fintech startup.
[Semantic] Alice prefers concise answers with code examples.
[Episodic] 2024-01-15: Discussed PostgreSQL migration strategy.
[Procedural] Communication style: direct, technical, no preamble.
</memory>
```

Format is configurable. The default XML-like wrapper is readable by all major LLMs.

---

### 8. Python SDK (`memory_vault/sdk/`)

High-level public interface. This is what end users `import`.

```python
from memory_vault import MemoryLayer

ml = MemoryLayer(user_id="...", config=MemoryConfig(...))
await ml.save(text, session_id="...")
context = await ml.recall(query, token_budget=1500)
await ml.forget(memory_id="...")
memories = await ml.list(memory_type="semantic")
```

**The SDK is the contract.** Anything in `memory_vault.sdk` is public API.
Breaking changes require a major version bump and `docs/api/API_SPEC.md` update.

---

### 9. CLI (`memory_vault/cli/`)

Debug and admin tooling built with Typer + Rich.

```bash
memory-vault memory list --user alice --type semantic
memory-vault memory search --user alice "PostgreSQL"
memory-vault memory delete --id <memory_id>
memory-vault session stats --user alice
memory-vault compress --user alice --dry-run
memory-vault server start --port 8000
```

---

### 10. MCP Server (`memory_vault/mcp/`)

Exposes memory operations as MCP tools, enabling direct integration with
Claude Code, Cursor, Windsurf, and any MCP-compatible AI tool.

**Exposed tools:**
- `memory_save` — save a memory chunk
- `memory_recall` — retrieve relevant memories
- `memory_list` — list all memories for a user
- `memory_forget` — delete a memory

**Guide:** See `docs/guides/MCP_INTEGRATION.md`.

---

## Data Models

Core Pydantic models live in `memory_vault/models.py`:

```python
class MemoryChunk(BaseModel):
    id: str
    user_id: str
    session_id: str
    content: str
    memory_type: MemoryType   # episodic | semantic | working | procedural
    importance: float          # 0.0 – 1.0
    embedding: list[float] | None
    created_at: datetime
    compressed: bool = False
    metadata: dict = {}

class RecallResult(BaseModel):
    memories: list[MemoryChunk]
    total_tokens: int
    budget_used: float         # 0.0 – 1.0

class MemoryConfig(BaseModel):
    token_budget: int = 2000
    top_k: int = 5
    compression_threshold: int = 10
    embedding_model: str = "all-MiniLM-L6-v2"
    storage_backend: Literal["chroma", "qdrant"] = "chroma"
```

---

## Configuration

All configuration flows through `memory_vault/config.py` using Pydantic Settings.
Environment variables override defaults. See `.env.example` for all options.

---

## Dependency Graph (no circular imports allowed)

```
cli, api, mcp, sdk
      │
   core engine (ingestion, retrieval, budget, prompt, compression)
      │
   storage (adapters)
      │
   models, config, exceptions, utils
```

`models`, `config`, `exceptions`, and `utils` must never import from higher layers.

---

## Architecture Decision Records

Key decisions are documented in `docs/adr/`:

| ADR | Decision |
|---|---|
| ADR-001 | Use ChromaDB as default vector store (not Qdrant) for embedded mode |
| ADR-002 | Use semantic chunking instead of fixed-size chunking |
| ADR-003 | Four memory types modeled after human memory research |
| ADR-004 | Async-first design using asyncio + anyio |
| ADR-005 | Token counting with tiktoken, not character proxies |
