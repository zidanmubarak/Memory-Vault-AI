# Configuration Guide — Memory Vault AI

> All configuration is driven by environment variables and the `MemoryConfig` Pydantic model.
> Environment variables always override `MemoryConfig` defaults.
> A `.env` file in the project root is loaded automatically in development.

---

## Quick Setup

```bash
cp .env.example .env
# Edit .env with your settings
```

---

## Minimal `.env` (Development)

```bash
# Uses ChromaDB + SQLite locally, no auth, no external services
ML_CHROMA_PATH=./data/chroma
ML_SQLITE_PATH=./data/memory.db
ML_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## Production `.env`

```bash
# Storage
ML_STORAGE_BACKEND=qdrant
ML_QDRANT_URL=http://qdrant:6333
ML_QDRANT_API_KEY=your-qdrant-cloud-key
ML_SQLITE_PATH=/var/lib/memory-vault/memory.db

# Security
ML_API_KEY=your-random-64-char-secret
ML_CORS_ORIGINS=https://yourapp.com,https://app2.com

# Embedding
ML_EMBEDDING_MODEL=all-MiniLM-L6-v2
ML_EMBEDDING_DEVICE=cpu          # or: cuda, mps

# Memory behavior
ML_DEFAULT_TOKEN_BUDGET=2000
ML_DEFAULT_TOP_K=5
ML_COMPRESSION_THRESHOLD=10
ML_IMPORTANCE_THRESHOLD=0.3
ML_RERANKER_ENABLED=false
ML_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Compression LLM (used for summarizing old episodic memories)
ML_COMPRESSION_MODEL=claude-haiku-4-5-20251001
ML_COMPRESSION_API_KEY=your-anthropic-key

# Server
ML_WORKERS=4
ML_PORT=8000
ML_LOG_LEVEL=INFO
ML_LOG_FORMAT=json
ML_LOG_SANITIZE=true             # Prevent memory content appearing in logs
```

---

## All Configuration Options

### Storage

| Variable | Type | Default | Description |
|---|---|---|---|
| `ML_STORAGE_BACKEND` | str | `chroma` | Vector store: `chroma` or `qdrant` |
| `ML_CHROMA_PATH` | str | `./data/chroma` | ChromaDB on-disk path |
| `ML_SQLITE_PATH` | str | `./data/memory.db` | SQLite file path |
| `ML_QDRANT_URL` | str | — | Qdrant server URL (if using Qdrant) |
| `ML_QDRANT_API_KEY` | str | — | Qdrant Cloud API key (optional for local) |
| `ML_QDRANT_COLLECTION` | str | `memory_vault` | Qdrant collection name |
| `ML_METADATA_BACKEND` | str | `sqlite` | Metadata store: `sqlite` or `postgres` |
| `ML_POSTGRES_URL` | str | — | PostgreSQL URL for multi-instance deployments |

### Embedding

| Variable | Type | Default | Description |
|---|---|---|---|
| `ML_EMBEDDING_MODEL` | str | `all-MiniLM-L6-v2` | Sentence-transformers model name |
| `ML_EMBEDDING_DEVICE` | str | `cpu` | `cpu`, `cuda`, or `mps` |
| `ML_EMBEDDING_BATCH_SIZE` | int | `32` | Chunks per embedding batch |
| `ML_EMBEDDING_CACHE` | bool | `true` | Cache embeddings by content hash |

**Available embedding models (trade-off: quality vs. speed):**

| Model | Dimensions | Speed | Quality | Best for |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | Fast | Good | Default — development + production |
| `all-mpnet-base-v2` | 768 | Medium | Better | When retrieval quality matters most |
| `multi-qa-MiniLM-L6-cos-v1` | 384 | Fast | Good | Query-optimized retrieval |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Medium | Good | Multilingual content |

### Memory Behavior

| Variable | Type | Default | Description |
|---|---|---|---|
| `ML_DEFAULT_TOKEN_BUDGET` | int | `2000` | Default max tokens per recall |
| `ML_DEFAULT_TOP_K` | int | `5` | Default memories returned per recall |
| `ML_IMPORTANCE_THRESHOLD` | float | `0.3` | Min importance score to save a chunk |
| `ML_COMPRESSION_THRESHOLD` | int | `10` | Sessions before auto-compression triggers |
| `ML_RERANKER_ENABLED` | bool | `false` | Enable cross-encoder re-ranking (adds latency) |
| `ML_RERANKER_MODEL` | str | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model used when reranker is enabled |
| `ML_MAX_CHUNK_TOKENS` | int | `300` | Max tokens per chunk during ingestion |
| `ML_MIN_CHUNK_TOKENS` | int | `50` | Min tokens per chunk (merge if smaller) |

### Compression

| Variable | Type | Default | Description |
|---|---|---|---|
| `ML_COMPRESSION_MODEL` | str | — | LLM for summarization (e.g. `claude-haiku-4-5-20251001`) |
| `ML_COMPRESSION_API_KEY` | str | — | API key for compression LLM |
| `ML_COMPRESSION_API_BASE` | str | — | Custom base URL (for OpenAI-compatible APIs) |
| `ML_COMPRESSION_SESSIONS` | int | `5` | Sessions to compress per job run |

### API Server

| Variable | Type | Default | Description |
|---|---|---|---|
| `ML_API_KEY` | str | — | Enables Bearer auth. Unset = no auth (dev only). |
| `ML_PORT` | int | `8000` | API server port |
| `ML_HOST` | str | `0.0.0.0` | Bind address |
| `ML_WORKERS` | int | `1` | Uvicorn worker count |
| `ML_CORS_ORIGINS` | str | `*` | Comma-separated allowed origins |
| `ML_RATE_LIMIT_SAVE` | int | `100` | Max save requests/min per user |
| `ML_RATE_LIMIT_RECALL` | int | `200` | Max recall requests/min per user |

Current enforcement scope:
- `ML_RATE_LIMIT_SAVE` applies to `POST /v1/memory`
- `ML_RATE_LIMIT_RECALL` applies to `GET /v1/memory/recall`

### Observability

| Variable | Type | Default | Description |
|---|---|---|---|
| `ML_LOG_LEVEL` | str | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `ML_LOG_FORMAT` | str | `text` | `text` or `json` |
| `ML_LOG_SANITIZE` | bool | `false` | Redact memory content from logs |
| `ML_METRICS_ENABLED` | bool | `false` | Enable Prometheus `/metrics` endpoint |

Metrics endpoint behavior:
- When `ML_METRICS_ENABLED=true`, `GET /metrics` returns Prometheus text exposition.
- When `ML_METRICS_ENABLED=false`, `GET /metrics` returns `404`.

---

## Using `MemoryConfig` in Code

```python
from memory_vault import MemoryLayer, MemoryConfig

config = MemoryConfig(
    token_budget=3000,
    top_k=8,
    embedding_model="all-mpnet-base-v2",
    storage_backend="qdrant",
    qdrant_url="http://localhost:6333",
    compression_threshold=5,
    reranker_enabled=True,
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
)

memory = MemoryLayer(user_id="alice", config=config)
```

Environment variables take precedence over `MemoryConfig` values.
`MemoryConfig` defaults take precedence over built-in defaults.

---

## `.env.example`

Copy this file to `.env` and fill in your values:

```bash
# ── Storage ──────────────────────────────────────
ML_STORAGE_BACKEND=chroma
ML_CHROMA_PATH=./data/chroma
ML_SQLITE_PATH=./data/memory.db
# ML_QDRANT_URL=http://localhost:6333
# ML_QDRANT_API_KEY=

# ── Embedding ────────────────────────────────────
ML_EMBEDDING_MODEL=all-MiniLM-L6-v2
ML_EMBEDDING_DEVICE=cpu

# ── Memory behavior ──────────────────────────────
ML_DEFAULT_TOKEN_BUDGET=2000
ML_DEFAULT_TOP_K=5
ML_IMPORTANCE_THRESHOLD=0.3
ML_COMPRESSION_THRESHOLD=10
ML_RERANKER_ENABLED=false
# ML_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# ── Compression LLM (optional) ───────────────────
# ML_COMPRESSION_MODEL=claude-haiku-4-5-20251001
# ML_COMPRESSION_API_KEY=

# ── API Server ───────────────────────────────────
# ML_API_KEY=                  # Leave unset for local dev
ML_PORT=8000
ML_WORKERS=1
ML_CORS_ORIGINS=*

# ── Logging ──────────────────────────────────────
ML_LOG_LEVEL=INFO
ML_LOG_FORMAT=text
ML_LOG_SANITIZE=false
```
