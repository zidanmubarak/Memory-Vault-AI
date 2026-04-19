# ADR-007: PostgreSQL as SQLite Replacement for Multi-Instance Deployments

**Date:** 2024-01  
**Status:** Proposed (not yet implemented)  
**Deciders:** Core team  
**Triggered by:** Kubernetes deployment pattern where multiple API replicas share state

---

## Context

SQLite works well for single-process deployments. When running multiple API server
instances (e.g. Kubernetes with 3+ replicas, or a multi-worker Gunicorn setup), all
instances must read and write to the same metadata store. SQLite does not support
concurrent writes from multiple processes reliably.

This ADR addresses the path forward for high-availability deployments.

## Decision

Support **PostgreSQL as an optional metadata backend**, replacing SQLite for
multi-instance deployments, while keeping SQLite as the default for local and
single-instance use.

**Implementation approach:**

1. `StorageLayer` already abstracts over the metadata backend via `SQLiteAdapter`
2. Add `PostgreSQLAdapter` implementing the same interface (`storage/postgres.py`)
3. Controlled by `ML_METADATA_BACKEND=postgres` environment variable
4. Connection: `ML_POSTGRES_URL=postgresql+asyncpg://user:pass@host/db`
5. Alembic migrations must support both backends — test on both in CI

**When to use PostgreSQL:**
- Running 2+ API server instances
- Kubernetes deployment
- Require ACID transactions across multiple memory operations

**When SQLite is fine:**
- Single server deployment
- Docker Compose with one API replica
- Local development

## Consequences

- Adds `asyncpg` and `psycopg2` as optional dependencies (`pip install memory-layer-ai[postgres]`)
- Alembic migration scripts must be tested against both SQLite and PostgreSQL
- `StorageLayer` init logic must branch on `ML_METADATA_BACKEND`
- CI pipeline must include a PostgreSQL service container for integration tests
- Documentation and deployment guide updated to include PostgreSQL option

## Status

This ADR is **proposed but not implemented**. Implementation is scheduled for v0.3.
Until then, multi-instance deployments should use a single instance behind a load balancer
or accept SQLite's concurrent-write limitations (low risk for low-traffic deployments).

## Rejected Alternatives

- **Distributed SQLite (Litestream, rqlite):** Adds infrastructure complexity comparable
  to PostgreSQL without the ecosystem benefits.
- **Redis for metadata:** Fast but not suitable as a primary metadata store (no relations,
  no complex queries, TTL-based storage doesn't fit permanent memory).
- **In-memory only:** Not durable. Defeats the purpose of persistent memory.
