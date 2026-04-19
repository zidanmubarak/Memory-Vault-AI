# ADR-004: Async-First Design Using asyncio + anyio

**Date:** 2024-01  
**Status:** Accepted  
**Deciders:** Core team

---

## Context

Memory Layer AI performs significant I/O: embedding model inference, vector DB queries,
SQLite reads/writes, and optional LLM calls for compression. We must decide whether the
core library is synchronous, asynchronous, or both.

## Decision

The core library is **async-first**: all public methods are `async def`, and all I/O
uses `await`. We use **anyio** as the async compatibility layer so the library works
with both asyncio and trio event loops.

**Rationale:**

Most modern Python frameworks that would integrate with Memory Layer AI are async:
FastAPI, Starlette, LangChain async, Anthropic Python SDK (async client), httpx.
A sync library in an async framework requires `asyncio.run()` in a thread or a sync
wrapper — both introduce overhead and complicate error handling.

Embedding computation with `sentence-transformers` is the one synchronous bottleneck.
We run it in a thread pool executor: `await asyncio.get_event_loop().run_in_executor(None, encode_fn, chunks)`.

anyio is chosen over raw asyncio because it provides a cleaner API for structured
concurrency and makes future trio compatibility possible without rewrites.

## Consequences

- Users integrating in a synchronous context must use `asyncio.run()` — we provide
  a sync wrapper `MemoryLayerSync` as a convenience class
- All internal I/O (ChromaDB, SQLite via aiosqlite, httpx for remote models) must
  use async-compatible drivers
- Background compression jobs use asyncio tasks, not threads
- Testing requires `pytest-asyncio` — all test fixtures for async code use `@pytest.mark.asyncio`

## Rejected Alternatives

- **Sync-first:** Natural for library code, but creates friction with async integrations.
  Embedding in FastAPI would require `run_in_executor` on every SDK call.
- **Both sync and async (parallel interfaces):** Doubles implementation surface. Not worth
  the maintenance cost for v0.x. The sync wrapper satisfies sync users.
