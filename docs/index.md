# Memory Layer AI Documentation

Welcome to the public docs portal for Memory Layer AI.
This site helps developers understand, integrate, deploy, and extend Memory Layer AI quickly.

## Start in 60 Seconds

1. Pick your path from the quick links below.
2. Follow the SDK guide for embedded use, or deployment guide for service mode.
3. Use API and spec docs when implementing production integrations.

## Quick Links

| Goal | Best page |
|---|---|
| Save and recall memory in Python | [SDK Guide](guides/SDK_GUIDE.md) |
| Configure runtime behavior | [Configuration](guides/CONFIGURATION.md) |
| Run with Docker or production topology | [Deployment](guides/DEPLOYMENT.md) |
| Integrate with coding assistants | [MCP Integration](guides/MCP_INTEGRATION.md) |
| Benchmark performance | [Benchmarking](guides/BENCHMARKING.md) |
| Extend memory classification | [Plugin System Guide](guides/PLUGIN_SYSTEM.md) |

## What You Can Do With Memory Layer AI

- Understand architecture and memory flow end-to-end.
- Integrate Memory Layer through SDK, REST API, CLI, or MCP.
- Configure storage and runtime behavior for local or production use.
- Extend memory routing using custom memory type plugins.

## Quick Start

```python
import asyncio

from memory_layer import MemoryLayer


async def main() -> None:
	memory = MemoryLayer(user_id="alice")
	await memory.save("I prefer concise answers with examples.")
	result = await memory.recall("How should I respond to the user?")
	print(result.prompt_block)


asyncio.run(main())
```

## Technical Reference

- [REST API Spec](api/API_SPEC.md)
- [SDK API Spec](api/SDK_API_SPEC.md)
- [Architecture](architecture/ARCHITECTURE.md)
- [Memory Logic](specs/MEMORY_LOGIC.md)
- [Database Schema](specs/DATABASE_SCHEMA.md)
- [Plugin System Spec](specs/PLUGIN_SYSTEM.md)

## Architecture Decisions

- Review ADR entries in the left navigation to understand design tradeoffs.

## Public Hosting via GitHub Pages

This docs site is designed for GitHub Pages deployment via GitHub Actions.
After Pages is enabled in repository settings, docs are available at:

- https://zidanmubarak.github.io/Memory-Layer-AI/


## Build This Website Locally

```bash
pip install -e ".[docs]"
python -m mkdocs serve
```

Static build output:

```bash
python -m mkdocs build
```
