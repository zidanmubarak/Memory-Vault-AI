# ADR-001: Use ChromaDB as default vector store

**Date:** 2024-01  
**Status:** Accepted  
**Deciders:** Core team

---

## Context

We need a vector database for storing and querying memory embeddings. Options considered:
Qdrant, Weaviate, Pinecone, ChromaDB, pgvector (via PostgreSQL).

The primary use case for v0.x is **local/embedded operation** — a developer should be able
to `pip install memory-layer-ai` and run without spinning up any external services.

## Decision

Use **ChromaDB** as the default storage backend.

**Rationale:**
- Embedded mode with zero external dependencies (pure Python, persists to disk)
- Simple Python API, easy to test and mock
- Sufficient performance for single-user local use (our MVP target)
- Clear migration path to Qdrant for production (we abstract behind `StorageBackend`)

## Consequences

- Qdrant adapter will be built in v0.3 for production use cases
- `StorageBackend` abstract base must be designed so both adapters are interchangeable
- ChromaDB's lack of native re-ranking means we implement our own in `retrieval/reranker.py`
- ChromaDB filtering is limited — complex metadata queries may require SQLite fallback

## Rejected Alternatives

- **Qdrant:** Requires a running server. Too heavy for local-first MVP.
- **pgvector:** Requires PostgreSQL. Adds operational complexity.
- **Pinecone:** Cloud-only, paid, no local option.
