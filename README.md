<p align="center">
  <img src="https://raw.githubusercontent.com/zidanmubarak/Memory-Vault-AI/main/imgs/banner.svg" alt="Memory Vault AI" width="800"/>
</p>

<p align="center">
  <a href="https://pypi.org/project/memory-vault/"><img src="https://img.shields.io/pypi/v/memory-vault?color=blue&label=PyPI" alt="PyPI version"/></a>
  <a href="https://pypi.org/project/memory-vault/"><img src="https://img.shields.io/pypi/pyversions/memory-vault" alt="Python versions"/></a>
  <a href="https://github.com/zidanmubarak/Memory-Vault-AI/blob/main/LICENSE"><img src="https://img.shields.io/github/license/zidanmubarak/Memory-Vault-AI?color=green" alt="License"/></a>
  <a href="https://github.com/zidanmubarak/Memory-Vault-AI/stargazers"><img src="https://img.shields.io/github/stars/zidanmubarak/Memory-Vault-AI?style=social" alt="Stars"/></a>
</p>

<p align="center">
  <b>Give any AI model persistent, context-aware memory — across every session.</b>
</p>

<p align="center">
  <a href="https://zidanmubarak.github.io/Memory-Vault-AI/">Documentation</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-four-modes">Four Modes</a> •
  <a href="https://pypi.org/project/memory-vault/">PyPI</a>
</p>

---

Memory Vault AI sits **between a user and any LLM**. It captures conversation history, extracts important facts, compresses old context, and injects the most relevant memories into each new prompt — automatically.

The result: an AI assistant that **remembers** who you are, what you've discussed, and how you prefer to communicate.

## ⚡ Quick Start

Install with pip:

```bash
pip install memory-vault
```

Then use it in 5 lines of Python:

```python
import asyncio
from memory_vault import MemoryLayer

async def main():
    memory = MemoryLayer(user_id="alice")

    await memory.save("My name is Alice, I'm a backend engineer.")
    await memory.save("I prefer dark mode and concise answers with code.")

    result = await memory.recall("What does the user prefer?")
    print(result.prompt_block)

asyncio.run(main())
```

That's it. No server, no config files, no external database. Just `pip install` and go.

---

## 🧠 Why Memory Vault AI?

| Problem | Memory Vault AI |
|---|---|
| 🔄 AI forgets everything between sessions | Persistent memory across unlimited sessions |
| 📦 Context window fills up with raw history | Smart compression + retrieval, not history dump |
| 🔒 Tied to one model or cloud vendor | Model-agnostic: Claude, GPT, Ollama, Gemini, any LLM |
| 👁️ No privacy control | User-scoped, full delete support, local-first |
| 🗃️ Black-box memory | Fully introspectable: view and edit what the system knows |

---

## ✨ Features

- **Four memory types** — episodic, semantic, working, procedural
- **Semantic retrieval** — vector similarity search, not keyword matching
- **Token-budget-aware** — never overflows your LLM's context window
- **Auto-compression** — old memories summarized, not deleted
- **Local-first** — runs fully offline with ChromaDB + local embeddings
- **MCP-compatible** — plug directly into Claude Desktop, Cursor, and any MCP tool
- **REST API + Python SDK + CLI + MCP** — four modes for every use case
- **Custom memory type plugins** — extend routing with pluggable classifiers

---

## 🎯 Four Modes

Memory Vault AI gives you **four ways** to interact with the memory system, each designed for a specific workflow:

| Mode | Best for | Command |
|---|---|---|
| **SDK** | Embedding memory into your Python app | `from memory_vault import MemoryLayer` |
| **API** | Multi-service architecture, any language | `uvicorn memory_vault.api.main:app` |
| **CLI** | Debugging, maintenance, inspection | `memory-vault memory list` |
| **MCP** | AI agent integration (Claude, Cursor) | `memory-vault mcp start` |

---

### 📦 Mode 1 — SDK (Python Library)

Use Memory Vault as an embedded library. No server needed.

<p align="center">
  <img src="https://raw.githubusercontent.com/zidanmubarak/Memory-Vault-AI/main/imgs/sdk-demo.svg" alt="SDK Demo" width="800"/>
</p>

```python
import asyncio
from memory_vault import MemoryLayer

async def main():
    memory = MemoryLayer(user_id="alice")

    # Save memories (auto-classified into episodic/semantic/procedural)
    await memory.save("My name is Alice, I'm a backend engineer.")
    await memory.save("I prefer concise answers with code examples.")
    await memory.save("I'm building a FastAPI app using PostgreSQL.")

    # Recall the most relevant memories for a query
    result = await memory.recall("What database is the user using?")
    print(result.prompt_block)   # Ready to inject into any LLM prompt
    print(result.total_tokens)   # Token count for budget tracking

asyncio.run(main())
```

<details>
<summary>More SDK examples</summary>

**Session management:**
```python
# Start a tracked session
session = await memory.start_session(metadata={"topic": "database design"})

# End session (working memory is archived)
await memory.end_session(session.session_id)
```

**Memory compression:**
```python
# Compress old episodic memories into summaries
result = await memory.compress(strategy="summarize")
print(f"Compressed {result.original_count} → {result.compressed_count} memories")
```

**Pagination & filtering:**
```python
# List all semantic memories
memories = await memory.list_memories(memory_type="semantic", page=1, page_size=10)
```

</details>

---

### 🌐 Mode 2 — REST API

Run Memory Vault as a standalone API server.

<p align="center">
  <img src="https://raw.githubusercontent.com/zidanmubarak/Memory-Vault-AI/main/imgs/api-demo.svg" alt="API Demo" width="800"/>
</p>

```bash
# Start the API server
uvicorn memory_vault.api.main:app --port 8000

# Save a memory
curl -X POST http://localhost:8000/v1/memory \
  -H "Content-Type: application/json" \
  -d '{"user_id": "alice", "content": "I prefer dark mode UI."}'

# Recall memories
curl "http://localhost:8000/v1/memory/recall?user_id=alice&query=preferences&top_k=3"
```

**Available endpoints:**

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/memory` | Save a new memory |
| `GET` | `/v1/memory/recall` | Recall relevant memories |
| `GET` | `/v1/memory/list` | List memories with filtering |
| `DELETE` | `/v1/memory/{id}` | Delete a specific memory |
| `POST` | `/v1/session/start` | Start a tracked session |
| `POST` | `/v1/session/end` | End a session |
| `GET` | `/v1/health` | Health check |
| `GET` | `/ui` | Built-in web introspection UI |
| `GET` | `/docs` | Interactive Swagger docs |

**Or run with Docker Compose (API + Qdrant):**

```bash
docker compose up --build
```

---

### 🖥️ Mode 3 — CLI

Inspect, search, and manage memories from the terminal.

<p align="center">
  <img src="https://raw.githubusercontent.com/zidanmubarak/Memory-Vault-AI/main/imgs/cli-demo.svg" alt="CLI Demo" width="800"/>
</p>

```bash
# List memories for a user
memory-vault memory list --user-id=alice --limit 10

# Semantic search
memory-vault search "What does the user prefer?" --user-id=alice

# Compress old memories
memory-vault compress --user-id=alice --strategy=summarize

# Show system stats
memory-vault stats

# Delete a specific memory
memory-vault delete d4a2-4f3b --user-id=alice
```

---

### 🤖 Mode 4 — MCP (Model Context Protocol)

Connect Memory Vault directly to AI agents like **Claude Desktop**, **Cursor**, or any MCP-enabled tool.

<p align="center">
  <img src="https://raw.githubusercontent.com/zidanmubarak/Memory-Vault-AI/main/imgs/mcp-demo.svg" alt="MCP Demo" width="800"/>
</p>

**Claude Desktop configuration** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "memory-vault": {
      "command": "memory-vault",
      "args": ["mcp", "start"],
      "env": {
        "ML_STORAGE_BACKEND": "chroma",
        "ML_DEFAULT_TOKEN_BUDGET": "2000"
      }
    }
  }
}
```

Once connected, your AI agent can automatically save and recall memories using the 6 built-in MCP tools.

---

## 🏗️ Architecture

<p align="center">
  <img src="https://raw.githubusercontent.com/zidanmubarak/Memory-Vault-AI/main/imgs/architecture.svg" alt="Architecture" width="800"/>
</p>

Memory Vault uses a **four-type memory system** inspired by cognitive science:

| Memory Type | What it stores | Retention |
|---|---|---|
| **Episodic** | Raw conversation turns, timestamped | Compressed after N sessions |
| **Semantic** | Extracted facts about the user | Permanent until deleted |
| **Working** | Current session context | Session-scoped, cleared on end |
| **Procedural** | User preferences, tone, habits | Permanent, updated over time |

Full architecture documentation: [ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md)

---

## 📦 Installation

<p align="center">
  <img src="https://raw.githubusercontent.com/zidanmubarak/Memory-Vault-AI/main/imgs/install.svg" alt="Installation" width="600"/>
</p>

```bash
# Recommended — includes ChromaDB, SQLite, and local embeddings
pip install memory-vault

# Add Qdrant support for production
pip install "memory-vault[qdrant]"

# Add MCP server for AI agent integration
pip install "memory-vault[mcp]"

# Everything (Qdrant + Postgres + MCP)
pip install "memory-vault[all]"
```

**Requirements:** Python 3.11+

---

## ⚙️ Configuration

Memory Vault works with zero config out of the box. For customization, use environment variables:

```bash
# Storage
ML_STORAGE_BACKEND=chroma          # or "qdrant"
ML_CHROMA_PATH=./data/chroma       # ChromaDB persistence path
ML_SQLITE_PATH=./data/memory.db    # SQLite metadata path

# AI/Embeddings
ML_EMBEDDING_MODEL=all-MiniLM-L6-v2   # Sentence-transformers model

# Memory
ML_DEFAULT_TOKEN_BUDGET=2000       # Max tokens for injected memory
ML_COMPRESSION_THRESHOLD=10        # Sessions before compression
```

Full configuration guide: [CONFIGURATION.md](docs/guides/CONFIGURATION.md)

---

## 📚 Documentation

| Document | Description |
|---|---|
| [**SDK Guide**](docs/guides/SDK_GUIDE.md) | Using Memory Vault as a Python library |
| [**API Reference**](docs/api/API_SPEC.md) | REST endpoint contracts |
| [**MCP Integration**](docs/guides/MCP_INTEGRATION.md) | Connecting to Claude, Cursor, etc. |
| [**Architecture**](docs/architecture/ARCHITECTURE.md) | System design and data flow |
| [**Memory Logic**](docs/specs/MEMORY_LOGIC.md) | Ingestion, retrieval, and compression algorithms |
| [**Plugin System**](docs/guides/PLUGIN_SYSTEM.md) | Building custom memory type classifiers |
| [**Deployment**](docs/guides/DEPLOYMENT.md) | Docker, systemd, Kubernetes guides |
| [**Benchmarking**](docs/guides/BENCHMARKING.md) | Performance benchmark suite |

📖 **Full docs website:** [zidanmubarak.github.io/Memory-Vault-AI](https://zidanmubarak.github.io/Memory-Vault-AI/)

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Language** | Python 3.11+ |
| **Framework** | FastAPI + Pydantic v2 |
| **Vector Store** | ChromaDB (local) / Qdrant (production) |
| **Metadata** | SQLite via SQLModel |
| **Embeddings** | sentence-transformers (local) |
| **CLI** | Typer + Rich |
| **Async** | anyio / asyncio |
| **MCP** | Model Context Protocol SDK |

---

## 🗺️ Roadmap

- [x] Core ingestion + retrieval engine
- [x] Four memory types (episodic, semantic, working, procedural)
- [x] REST API with FastAPI
- [x] CLI tools with Rich tables
- [x] MCP server integration
- [x] Custom memory type plugins
- [x] Auto-compression engine
- [x] Published on PyPI
- [ ] Multi-user auth & API keys
- [ ] PostgreSQL metadata backend
- [ ] Cloud-hosted managed service
- [ ] LangChain / LlamaIndex integrations
- [ ] Memory import/export

---

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Clone and install for development
git clone https://github.com/zidanmubarak/Memory-Vault-AI
cd Memory-Vault-AI
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

<p align="center">
  <sub>Built with ❤️ for the AI community</sub>
</p>
