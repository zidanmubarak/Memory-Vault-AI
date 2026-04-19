# ADR-006: Single ChromaDB Collection with user_id Metadata Filter

**Date:** 2024-01  
**Status:** Accepted  
**Deciders:** Core team

---

## Context

When storing memories for multiple users in ChromaDB, we must choose between:
- One collection per user (`memory_alice`, `memory_bob`, ...)
- One global collection with `user_id` as a metadata filter field

## Decision

Use a **single global collection** named `memory_layer` with `user_id` stored as
document metadata, filtered at query time via `where={"user_id": user_id}`.

**Rationale:**

Per-user collections create operational complexity: collections must be created on first
use and deleted on user account deletion. With 10,000 users, ChromaDB manages 10,000
collections — a pattern its authors do not recommend at scale.

A single collection with metadata filtering is simpler to manage, easier to back up,
and how ChromaDB is designed to be used at scale. User isolation is enforced at the
query layer by always including `where={"user_id": ...}` — this must never be omitted.

**Risk mitigation:** The `StorageLayer` abstraction enforces that `user_id` is always
set in the metadata and always included in queries. Direct ChromaDB access is prohibited
in feature code — all access goes through `StorageLayer`.

## Consequences

- `StorageLayer.query()` must always accept and enforce `user_id` — it is a required
  parameter, never optional
- A missing `user_id` filter would return memories from all users — this is the primary
  security risk and is prevented by making it a required parameter with type enforcement
- Collection backup is a single operation, not per-user
- ChromaDB metadata filters have limited expressiveness — complex queries may require
  a SQLite fallback for metadata-only operations

## Rejected Alternatives

- **Per-user collections:** Simpler isolation model but poor operational scalability.
  ChromaDB collection overhead is non-trivial at thousands of users.
- **Per-memory-type collections (`episodic`, `semantic`, ...):** Fewer collections but
  queries across types require multiple round trips. Memory type filtering via metadata
  is sufficient and already supported.
