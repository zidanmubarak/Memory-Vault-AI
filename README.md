# Memory Vault AI

**Persistent, context-aware memory for any AI assistant.**

Memory Vault AI is an open-source Python library and REST API that gives AI models
the ability to remember users across sessions — automatically, efficiently, and with full
privacy control.

```python
import asyncio

from memory_vault import MemoryLayer

async def main() -> None:
	memory = MemoryLayer(user_id="user-123")

	# Save a conversation turn
	await memory.save("I'm building a FastAPI app using PostgreSQL")

	# Later — in a new session — recall relevant context
	result = await memory.recall("What database is the user using?")
	print(result.prompt_block)

asyncio.run(main())
```

---

## Why Memory Vault AI?

| Problem | Memory Vault AI |
|---|---|
| AI forgets everything between sessions | Persistent memory across unlimited sessions |
| Context window fills up with history | Smart compression and retrieval, not raw history dump |
| Tied to one model or cloud vendor | Model-agnostic: Claude, GPT, Ollama, any LLM |
| No privacy control | User-scoped memory, full delete support, local-first option |
| Black-box memory | Introspectable: view and edit what the system knows |

---

## Features

- **Four memory types** — episodic, semantic, working, procedural
- **Semantic retrieval** — vector similarity search, not keyword matching
- **Token-budget-aware** — never overflows your LLM's context window
- **Auto-compression** — old memories summarized, not deleted
- **Local-first** — runs fully offline with ChromaDB + local embedding models
- **MCP-compatible** — plug directly into Claude, Cursor, and any MCP-enabled tool
- **REST API + Python SDK** — use as a service or import as a library
- **CLI debug tools** — inspect, search, compress, and manage memories from the terminal
- **Custom memory type plugins** — extend ingestion routing with pluggable classifiers

---

## Quick Start

```bash
pip install "memory-vault[all]"
```

```python
import asyncio

from memory_vault import MemoryLayer

async def main() -> None:
	# Embedded mode (no server needed)
	memory = MemoryLayer(user_id="alice")

	await memory.save("My name is Alice, I'm a backend engineer.")
	await memory.save("I prefer concise answers with code examples.")

	context = await memory.recall("Tell me about the user")
	print(context.prompt_block)

asyncio.run(main())
```

**Or run as an API server:**

```bash
uvicorn memory_vault.api.main:app --port 8000
```

Memory introspection UI is available at `http://localhost:8000/ui`.

**Or run with Docker Compose (API + Qdrant):**

```bash
docker compose up --build
```

Then open:

- `http://localhost:8000/v1/health`
- `http://localhost:8000/docs`
- `http://localhost:8000/ui`

---

## Documentation

### Documentation Website

Build and run the docs as a simple website:

```bash
pip install -e ".[docs]"
python -m mkdocs serve
```

Then open `http://127.0.0.1:8000`.

To build static HTML output:

```bash
python -m mkdocs build
```

### Publish Documentation to GitHub Pages

This repository includes an automated Pages workflow at
[.github/workflows/docs-pages.yml](.github/workflows/docs-pages.yml).

To make docs publicly accessible without local setup:

1. Push to branch `main`.
2. Open GitHub repository settings.
3. Go to Pages.
4. Set source to `GitHub Actions`.
5. Wait for workflow `Docs Pages` to complete.

Public URL format:

- `https://zidanmubarak.github.io/Memory-Vault-AI/`

| Document | Description |
|---|---|
| [Architecture](docs/architecture/ARCHITECTURE.md) | System design and component overview |
| [Memory Logic](docs/specs/MEMORY_LOGIC.md) | How ingestion, retrieval and compression work |
| [API Reference](docs/api/API_SPEC.md) | REST endpoint contracts |
| [SDK Guide](docs/guides/SDK_GUIDE.md) | Using Memory Vault as a Python library |
| [MCP Integration](docs/guides/MCP_INTEGRATION.md) | Connecting to Claude Code, Cursor, etc. |
| [Benchmarking Guide](docs/guides/BENCHMARKING.md) | Running performance benchmark suite and baselines |
| [Plugin System Guide](docs/guides/PLUGIN_SYSTEM.md) | Building and registering custom memory type plugins |

---

## Tech Stack

- **Python 3.11+** · FastAPI · Pydantic v2
- **ChromaDB** (local) / **Qdrant** (production)
- **SQLite** via SQLModel for metadata
- **sentence-transformers** for local embeddings
- **Typer + Rich** for CLI
- **AsyncIO** for background jobs

---

## Project Status

**v0.1 — Active development.** Core ingestion and retrieval being built.
See milestone tracking in the GitHub repository issues/projects.

---

## Contributing

Contributions are welcome. Use GitHub Issues and Pull Requests in this repository.

---

## License

MIT — see [LICENSE](LICENSE).
