# ADR-002: Semantic Chunking over Fixed-Size Chunking

**Date:** 2024-01  
**Status:** Accepted  
**Deciders:** Core team

---

## Context

When ingesting text into memory, we must split it into storable units ("chunks").
Two primary strategies exist: fixed-size chunking (split every N tokens) and
semantic chunking (split at natural content boundaries).

## Decision

Use **semantic chunking** — split at double-newlines first, then at sentence boundaries
for long paragraphs, then merge short adjacent chunks.

**Rationale:**

Fixed-size chunking is simpler to implement but produces semantically incoherent chunks.
A sentence like "My preferred database is PostgreSQL. I use it for all projects." might
be split as "My preferred database is Postgre" + "SQL. I use it for all projects." —
destroying the semantic unit.

Semantic chunks produce self-contained, meaningful units that embed better (the embedding
model captures a complete thought, not a fragment), retrieve more accurately (the retrieved
chunk actually answers the query), and are human-readable in debug output.

**Bounds:** Min 50 tokens, max 300 tokens. Chunks below the minimum are merged with neighbors.
Chunks above the maximum are split at sentence boundaries.

## Consequences

- `SemanticChunker` is more complex to implement than a simple `text[:n]` split
- Chunk count per document is variable — context budget enforcement becomes more important
- Edge case: very long single sentences (> 300 tokens, e.g. code) are split at whitespace as fallback

## Rejected Alternatives

- **Fixed-size (N tokens):** Simple but semantically incoherent. Fragments entities and facts.
- **Sentence-only:** Produces too many small chunks for long documents, inflating vector DB size.
- **Paragraph-only:** Paragraphs vary too widely in size (1 sentence to 50+). Unreliable bounds.
