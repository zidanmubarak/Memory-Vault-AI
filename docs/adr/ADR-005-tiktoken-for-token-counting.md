# ADR-005: Token Counting with tiktoken, Not Character Proxies

**Date:** 2024-01  
**Status:** Accepted  
**Deciders:** Core team

---

## Context

The Context Budget Manager must enforce token limits so injected memory never overflows
an LLM's context window. Token counting can be approximated (characters / 4) or done
precisely with a tokenizer library.

## Decision

Use **tiktoken** with the `cl100k_base` encoding (used by GPT-4, Claude, and most modern
models) for all token counting.

**Rationale:**

Character-based approximations (chars / 4) have up to 30% error on code, multilingual
text, and special characters. An error of 30% on a 2000-token budget means we might inject
2600 tokens — enough to overflow smaller context windows or degrade output quality.

tiktoken is fast (Rust-backed), reliable, and supports all encodings used by major LLMs.
The `cl100k_base` encoding is a safe default that closely approximates token counts for
Claude, GPT-4, Mistral, and Llama 3.

For models using different tokenizers (e.g. Gemini), the count will be approximate but
within 5–10% — acceptable for budget enforcement purposes.

## Consequences

- `tiktoken` is a required dependency (adds ~2MB to install)
- Token counting is synchronous but fast (<1ms for typical chunks)
- The `token_budget` parameter in all public APIs refers to tiktoken `cl100k_base` tokens
- Users targeting models with very different tokenizers (unusual) may need to set a
  conservative budget (e.g. 80% of the actual limit)

## Rejected Alternatives

- **chars / 4 approximation:** Fast and dependency-free but 30% error is unacceptable
  for reliable budget enforcement.
- **Model-specific tokenizers:** More accurate, but requires knowing the target model
  upfront and managing multiple tokenizer dependencies. Not worth complexity for v0.x.
