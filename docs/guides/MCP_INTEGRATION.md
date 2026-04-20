# MCP Integration Guide — Memory Vault AI

> **Audience:** Developers who want to connect Memory Vault AI to AI coding tools
> like Claude Code, Cursor, Windsurf, or any MCP-compatible client.
>
> **What you get:** Your AI assistant gains persistent memory across every session —
> it remembers your stack, your conventions, your preferences, and what you've discussed.

---

## What is MCP?

Model Context Protocol (MCP) is an open standard that lets AI tools connect to external
data sources and services. Memory Vault AI exposes an MCP server that makes memory
operations available as tools any compatible AI client can call.

**Supported clients:** Claude Code, Cursor, Windsurf, Continue, and any client implementing MCP.

---

## Prerequisites

```bash
# Install Memory Vault AI with MCP support
pip install "memory-vault[mcp]"

# Verify installation
memory-vault mcp tools
```

---

## Starting the MCP Server

```bash
# Start with default settings (SQLite + ChromaDB, local embeddings)
memory-vault mcp start

# With custom config
memory-vault mcp start \
  --host 127.0.0.1 \
  --port 8001 \
  
```

The server starts an MCP-compatible endpoint at `http://localhost:8001/mcp/v1`.

Supported JSON-RPC methods at this endpoint:

- `initialize`
- `tools/list`
- `tools/call`

---

## Connecting Claude Code

### Option A: Project-level config (recommended)

Create `.claude/mcp.json` in your project root:

```json
{
  "mcpServers": {
    "memory-vault": {
      "url": "http://localhost:8001/mcp/v1",
      "description": "Persistent memory across sessions"
    }
  }
}
```

Claude Code will automatically detect and connect to this server when opened in the project.

### Option B: Global config

Add to `~/.claude/mcp.json` to enable memory for all projects:

```json
{
  "mcpServers": {
    "memory-vault": {
      "url": "http://localhost:8001/mcp/v1"
    }
  }
}
```

### Verify the connection

In Claude Code, type:

```
/mcp list
```

You should see `memory-vault` in the connected servers list with its available tools.

---

## Connecting Cursor

Add to Cursor settings (`Cursor > Settings > MCP Servers`):

```json
{
  "memory-vault": {
    "url": "http://localhost:8001/mcp/v1",
    "enabled": true
  }
}
```

Or edit `~/.cursor/mcp.json` directly:

```json
{
  "servers": [
    {
      "name": "memory-vault",
      "url": "http://localhost:8001/mcp/v1"
    }
  ]
}
```

---

## Connecting Windsurf

In Windsurf's MCP configuration panel, add:

| Field | Value |
|---|---|
| Name | `memory-vault` |
| URL | `http://localhost:8001/mcp/v1` |
| Transport | `http` |

---

## Available MCP Tools

Once connected, the following tools are available to the AI client:

### `memory_save`

Save a piece of information to persistent memory.

```
Tool: memory_save
Input:
  text: string           — the content to remember
  user_id: string        — user or project identifier
  session_id: string     — current session identifier
  memory_type: string?   — optional hint: episodic|semantic|procedural|working
```

### `memory_recall`

Retrieve memories relevant to a query.

```
Tool: memory_recall
Input:
  query: string          — natural language query
  user_id: string        — user or project identifier
  top_k: int?            — max memories to return (default: 5)
  token_budget: int?     — max tokens (default: 2000)
  memory_types: string?  — comma-separated filter (default: all)

Output:
  prompt_block: string   — formatted memory block ready to use
  memories: array        — individual memory chunks with metadata
  total_tokens: int
```

### `memory_list`

List stored memories for a user.

```
Tool: memory_list
Input:
  user_id: string
  memory_type: string?   — filter by type
  page: int?
  page_size: int?
```

### `memory_forget`

Delete a specific memory or all memories for a user.

```
Tool: memory_forget
Input:
  user_id: string
  memory_id: string?     — delete one. If omitted + confirm=true: delete all.
  confirm: bool?
```

---

## Recommended System Prompt

Add this to your AI client's system prompt (or project instructions) to activate
memory usage on every request:

```
You have access to a persistent memory system via MCP tools.

At the START of every session:
1. Call memory_recall with query="user background, current projects, preferences" 
   and user_id="<your_user_id>"
2. Use the returned context to personalize your responses

During our conversation:
- When you learn something important about me, my project, or my preferences,
  call memory_save to store it
- Important things to save: tech stack decisions, architectural choices,
  personal preferences, ongoing task context, errors encountered and their fixes

At the END of a session:
- Summarize key decisions made and call memory_save to store them

Never ask me to repeat context I've shared before. Check your memory first.
```

**For Claude Code specifically**, add this to `.claude/CLAUDE.md` in your project:

```markdown
## Memory Instructions

You have access to memory-vault MCP tools. On every new session:
1. Call `memory_recall` with the current task as query, user_id="<project_name>"
2. Store any new architectural decisions, conventions, or debugging findings with `memory_save`

Project memory user_id: "my-project-name"
```

---

## User ID Strategy

Choose a `user_id` strategy that fits your use case:

| Use case | Recommended `user_id` |
|---|---|
| Personal dev machine, single user | `"default"` or your username |
| Per-project memory | `"project-my-saas"` |
| Team with shared memory | `"team-backend"` |
| Multi-tenant app | `"user-{uuid}"` per user |

Keep `user_id` consistent across sessions — it's the key that connects all memories.

---

## Auto-start on Boot

### macOS (launchd)

Create `~/Library/LaunchAgents/ai.memoryvault.mcp.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>ai.memoryvault.mcp</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/local/bin/memory-vault</string>
    <string>mcp</string>
    <string>start</string>
    <string>--daemon</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>/tmp/memory-vault-mcp.log</string>
  <key>StandardErrorPath</key>
  <string>/tmp/memory-vault-mcp.err</string>
</dict>
</plist>
```

```bash
launchctl load ~/Library/LaunchAgents/ai.memoryvault.mcp.plist
```

### Linux (systemd)

Create `/etc/systemd/user/memory-vault-mcp.service`:

```ini
[Unit]
Description=Memory Vault AI MCP Server
After=network.target

[Service]
ExecStart=/usr/local/bin/memory-vault mcp start --daemon
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

```bash
systemctl --user enable memory-vault-mcp
systemctl --user start memory-vault-mcp
```

---

## Security Considerations

- **Local only by default:** The MCP server binds to `localhost` only. It is not exposed to the network.
- **No authentication in dev mode:** For production or shared machines, set `ML_API_KEY` in `.env` and the server will require `Authorization: Bearer <key>` on all MCP requests.
- **Memory isolation:** All memory is scoped to `user_id`. One user cannot access another's memories.
- **No data sent externally:** Embeddings are computed locally using `sentence-transformers`. Nothing is sent to external APIs unless you configure a remote embedding model.

---

## Troubleshooting

**MCP server not showing in client:**
```bash
# Check server is running
curl http://localhost:8001/mcp/v1/health

# List registered tools from local CLI
memory-vault mcp tools
```

**Memory not persisting between sessions:**
- Verify `chroma_path` and `sqlite_path` point to the same directory across restarts.
- Check disk write permissions on the data directory.
- Run `memory-vault memory list --user <user_id>` to confirm data exists.

**Slow recall (>500ms):**
- Disable reranker: `ML_RERANKER_ENABLED=false`
- Reduce `top_k` to 3
- Check embedding model is cached (first run downloads ~25MB)

**Tool call failing with 422:**
- Ensure `user_id` is provided in every tool call — it is always required.
- Check `memory_types` filter uses valid values: `episodic`, `semantic`, `working`, `procedural`.
