# Memory Logic Specification

> **Audience:** AI coding agents and contributors implementing core engine components.
> This document is the authoritative spec for all memory algorithms.
> Implementation must match this spec. If you disagree with the spec, open an ADR.

---

## 1. Ingestion Pipeline

### 1.1 Input

```python
async def ingest(
    text: str,
    user_id: str,
    session_id: str,
    memory_type_hint: MemoryType | None = None,
) -> list[MemoryChunk]
```

### 1.2 Step 1 — Text Cleaning

- Strip leading/trailing whitespace
- Normalize Unicode to NFC
- Collapse multiple newlines to double newline
- Do NOT strip formatting markers or code blocks

### 1.3 Step 2 — Semantic Chunking

Split text into chunks at semantic boundaries. Do not use fixed character or token counts.

**Algorithm:**
1. Split on double newlines (`\n\n`) as primary boundary
2. For chunks > `MAX_CHUNK_TOKENS` (default: 300), split further at sentence boundaries (`.`, `!`, `?` followed by space + capital)
3. Merge adjacent chunks shorter than `MIN_CHUNK_TOKENS` (default: 50) with their neighbor
4. Result: list of strings, each 50–300 tokens

**Why not fixed-size chunking?** See `docs/adr/ADR-002-chunking-strategy.md`.

### 1.4 Step 3 — Embedding

```python
embeddings = await embedder.encode_batch(chunks)
# Returns: list[list[float]], shape [n_chunks, embedding_dim]
```

- Use `sentence-transformers/all-MiniLM-L6-v2` by default (384 dimensions)
- Batched calls only — never embed one chunk at a time in a loop
- Cache embeddings by content hash to avoid re-embedding identical text

### 1.5 Step 4 — Importance Scoring

Score each chunk from 0.0 (skip) to 1.0 (high importance).

**Formula:**
```
importance = (novelty_score * 0.6) + (salience_score * 0.4)
```

**Novelty score:** Cosine distance between chunk embedding and the centroid of the user's
existing memory embeddings. High distance = novel = high novelty.
- If user has no existing memories: novelty = 1.0 for all chunks
- Clamp to [0.0, 1.0]

**Salience score:** Heuristic based on content signals:
- Contains named entity (person, org, place, product) → +0.3
- Contains number or date → +0.2
- Contains preference signal ("I prefer", "I always", "I hate", "my favorite") → +0.4
- Contains technical term (detected by average token rarity) → +0.1
- Maximum possible salience: 1.0

**Threshold:** Chunks with `importance < 0.3` are discarded (not stored).

### 1.6 Step 5 — Memory Type Routing

If `memory_type_hint` is provided, use it. Otherwise, classify:

| Rule | Memory Type |
|---|---|
| Contains preference signal | `procedural` |
| Explicitly about "now" / current session | `working` |
| Extractable as subject-predicate-object fact | `semantic` |
| Default (conversation turn, event) | `episodic` |

Procedural memory is written to SQLite key-value store directly.
Other types are stored in ChromaDB with appropriate metadata.

---

## 2. Retrieval Pipeline

### 2.1 Input

```python
async def recall(
    query: str,
    user_id: str,
    top_k: int = 5,
    token_budget: int = 2000,
    memory_types: list[MemoryType] | None = None,  # None = all types
) -> RecallResult
```

### 2.2 Step 1 — Query Embedding

Embed the query text using the same model used during ingestion. This is critical —
mismatched embedding models produce garbage retrieval results.

### 2.3 Step 2 — ANN Search

```python
candidates = await chroma.query(
    query_embeddings=[query_embedding],
    n_results=top_k_candidates,  # default: top_k * 4
    where={"user_id": user_id},
)
```

Fetch `top_k * 4` candidates to allow for post-filtering and re-ranking.

### 2.4 Step 3 — Candidate Filtering

Remove candidates where:
- `importance < 0.2` (below minimum threshold)
- `compressed = True` AND a summary already exists (avoid duplicating)
- Memory type not in `memory_types` filter (if provided)

### 2.5 Step 4 — Re-ranking (Optional)

If `reranker_enabled = True`:
- Use cross-encoder model to score query-memory relevance pairs
- Replace cosine similarity scores with cross-encoder scores
- Sort descending by new score
- If cross-encoder model/dependency is unavailable, fall back to weighted score blending

If disabled (default), use cosine similarity scores as-is.

### 2.6 Step 5 — Token Budget Enforcement

```python
selected = []
tokens_used = 0
for chunk in sorted_candidates:
    chunk_tokens = count_tokens(chunk.content)
    if tokens_used + chunk_tokens <= token_budget:
        selected.append(chunk)
        tokens_used += chunk_tokens
    if len(selected) >= top_k:
        break
```

Always include procedural memories (user preferences) first — they are small and
critical for consistent behavior.

### 2.7 Step 6 — Result Assembly

Return `RecallResult(memories=selected, total_tokens=tokens_used, budget_used=tokens_used/token_budget)`.

---

## 3. Compression Algorithm

Triggered when: number of uncompressed episodic sessions for a user > `compression_threshold`.

### 3.1 Session Grouping

Group episodic memories by `session_id`. Sort sessions by `created_at` ascending.
Take the oldest N sessions where N = `sessions_to_compress` (default: `compression_threshold / 2`).

### 3.2 Summarization

For each selected session:
1. Concatenate all episodic memory content in chronological order
2. Call LLM with summarization prompt:
   ```
   Summarize the following conversation into 2-3 sentences capturing key facts,
   decisions, and topics discussed. Be specific and preserve names, numbers, and
   technical details. Do not include pleasantries.
   
   Conversation:
   {session_content}
   ```
3. Store summary as a new `semantic` memory with `source_session_id` and `compression_source=True`
4. Mark original episodic chunks as `compressed=True`

### 3.3 Archive Policy

Compressed episodic memories are NOT deleted. They are excluded from retrieval by default
but queryable with `include_compressed=True`. This preserves audit trail.

---

## 4. Prompt Assembly Format

```
<memory>
{for each memory in RecallResult.memories}
[{memory.memory_type.upper()}] {memory.content}
{endfor}
</memory>
```

**Example:**
```
<memory>
[SEMANTIC] User is a backend engineer specializing in Python and PostgreSQL.
[SEMANTIC] User is building a SaaS product with FastAPI and Stripe integration.
[PROCEDURAL] Communication style: direct, technical, no preamble.
[EPISODIC] 2024-01-20: Discussed database migration from MySQL to PostgreSQL.
[EPISODIC] 2024-01-18: User reported issue with Alembic autogenerate on enums.
</memory>
```

The `<memory>` XML wrapper is recognized by Claude, GPT-4, and most capable models
as structured system context. Do not change the format without updating this spec.
