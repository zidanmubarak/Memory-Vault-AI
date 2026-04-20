# Database Schema — Memory Vault AI

> Two backends work together: **ChromaDB** for vector search, **SQLite** for structured metadata.
> Never query them independently from feature code — always use `StorageLayer`.

---

## SQLite Schema (via SQLModel)

### Table: `memory_chunks`

Primary metadata store for all memory records.

```sql
CREATE TABLE memory_chunks (
    id              TEXT PRIMARY KEY,        -- "mem_" + uuid4 hex
    user_id         TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    memory_type     TEXT NOT NULL,           -- episodic|semantic|working|procedural
    content         TEXT NOT NULL,
    importance      REAL NOT NULL,           -- 0.0 – 1.0
    token_count     INTEGER NOT NULL,
    compressed      BOOLEAN DEFAULT FALSE,
    compression_source BOOLEAN DEFAULT FALSE,
    source_session_id TEXT,                  -- set when this is a compression summary
    chroma_id       TEXT,                    -- corresponding ChromaDB document id (NULL for procedural)
    created_at      DATETIME NOT NULL,
    updated_at      DATETIME NOT NULL,
    metadata        TEXT DEFAULT '{}'        -- JSON blob for extensibility
);

CREATE INDEX idx_memory_user       ON memory_chunks(user_id);
CREATE INDEX idx_memory_user_type  ON memory_chunks(user_id, memory_type);
CREATE INDEX idx_memory_session    ON memory_chunks(session_id);
CREATE INDEX idx_memory_compressed ON memory_chunks(user_id, compressed);
```

### Table: `procedural_memory`

Key-value store for user preferences and habits.

```sql
CREATE TABLE procedural_memory (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    key         TEXT NOT NULL,               -- e.g. "communication_style", "preferred_language"
    value       TEXT NOT NULL,               -- JSON value
    confidence  REAL DEFAULT 1.0,
    updated_at  DATETIME NOT NULL,
    source_chunk_id TEXT                     -- which memory_chunk triggered this
);

CREATE UNIQUE INDEX idx_proc_user_key ON procedural_memory(user_id, key);
```

### Table: `sessions`

```sql
CREATE TABLE sessions (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    started_at      DATETIME NOT NULL,
    last_activity   DATETIME NOT NULL,
    ended_at        DATETIME,
    compressed      BOOLEAN DEFAULT FALSE,
    memory_count    INTEGER DEFAULT 0,
    total_tokens    INTEGER DEFAULT 0
);

CREATE INDEX idx_sessions_user ON sessions(user_id);
```

### Table: `compression_jobs`

```sql
CREATE TABLE compression_jobs (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    status          TEXT NOT NULL,           -- queued|running|completed|failed
    sessions_compressed INTEGER DEFAULT 0,
    memories_created    INTEGER DEFAULT 0,
    error           TEXT,
    created_at      DATETIME NOT NULL,
    completed_at    DATETIME
);
```

---

## ChromaDB Collections

### Collection: `memory_{user_id}` (per-user, or global with user_id filter)

> **Decision:** One collection per user vs. one global collection with metadata filtering.
> We use **one global collection** with `user_id` in metadata for simpler ops.
> See `docs/adr/ADR-006-chroma-collection-strategy.md`.

**Document structure:**
```python
collection.add(
    ids=["mem_abc123"],
    embeddings=[[0.1, 0.2, ...]],    # 384-dim float32
    documents=["raw text content"],
    metadatas=[{
        "user_id": "user_123",
        "session_id": "sess_xyz",
        "memory_type": "episodic",
        "importance": 0.72,
        "compressed": False,
        "created_at": "2024-01-20T10:30:00Z",
    }]
)
```

**Query:**
```python
results = collection.query(
    query_embeddings=[query_vec],
    n_results=20,
    where={
        "user_id": "user_123",
        "compressed": False,
    }
)
```

**Note:** Procedural memory is NOT stored in ChromaDB — only in SQLite. It's retrieved
directly by key, not by similarity search.

---

## ID Conventions

| Entity | Format | Example |
|---|---|---|
| Memory chunk | `mem_` + 12 hex chars | `mem_a3f9b2c10d4e` |
| Session | `sess_` + 8 hex chars | `sess_9a3b1c2d` |
| User | Passed in by caller | `user_alice` or UUID |
| Compression job | `job_` + 8 hex chars | `job_f1e2d3c4` |

---

## Migration Strategy

Schema migrations use **Alembic**. Migration files are in `alembic/versions/`.

```bash
# Generate migration
alembic revision --autogenerate -m "add compression_jobs table"

# Apply migrations
alembic upgrade head

# Rollback one step
alembic downgrade -1
```

Never edit SQLite schema manually. All changes must go through Alembic.
