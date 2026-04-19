# ADR-003: Four Memory Types Modeled After Human Memory Research

**Date:** 2024-01  
**Status:** Accepted  
**Deciders:** Core team

---

## Context

We need to decide how to categorize stored memories. A flat, single-type memory store
is simple but loses important distinctions: a user's name is different from what they
said two minutes ago, and both are different from their communication preferences.

## Decision

Model memory after the established taxonomy from cognitive science: **episodic, semantic,
working, and procedural memory**.

| Type | Human analogy | What we store | Retrieval |
|---|---|---|---|
| Episodic | "What happened" | Conversation turns, events, decisions | Vector search |
| Semantic | "What I know" | Extracted facts, entities, knowledge | Vector search |
| Working | "Right now" | Current session context | Direct lookup |
| Procedural | "How I do things" | Preferences, habits, styles | Key-value lookup |

**Why this taxonomy?**

It maps naturally onto the actual structure of user information without requiring
complex classification rules. The distinction between "Alice said X" (episodic) and
"Alice is a backend engineer" (semantic) is intuitive and makes retrieval more targeted:
a question about user preferences retrieves from `procedural`, not `episodic`.

It also aligns with how humans think about memory, making the system's behavior
explainable to end users.

## Consequences

- `MemoryRouter` must classify chunks into one of four types — auto-classification
  uses heuristics (preference signals, entity detection, temporal signals)
- Procedural memory uses key-value storage (SQLite only), not vector search — it is
  always retrieved, not semantically searched
- Working memory is session-scoped and not persisted across sessions
- Adds complexity vs. a flat store, but significantly improves retrieval precision

## Rejected Alternatives

- **Single flat type:** Simple, but retrieval becomes noisier. "What does Alice prefer?" 
  returns conversation turns mixed with facts.
- **Two types (short-term / long-term):** Better than flat but loses the episodic/semantic
  distinction that is crucial for accurate retrieval.
- **Custom taxonomy:** Possible future extension via plugin system, but premature for v0.x.
