# API Specification — Memory Vault AI

> **Version:** 0.1  
> **Base URL:** `http://localhost:8000/v1`  
> **Auth:** Bearer token via `Authorization: Bearer <api_key>` header (optional in dev mode)
> - When `ML_API_KEY` is configured, all `/v1/*` routes except `/v1/health` require this header.
> - Missing/invalid token returns `401 Unauthorized`.
> **OpenAPI:** Swagger UI at `/docs`, schema JSON at `/openapi.json`.
> **UI:** Memory introspection page at `/ui`.
>
> **AI agents:** Do not change these endpoint signatures. If a feature requires
> a new endpoint or modified response shape, update this spec first, then implement.

---

## Endpoints

### POST /v1/memory

Save a memory chunk for a user.

**Request:**
```json
{
  "user_id": "string (required)",
  "session_id": "string (required)",
  "text": "string (required)",
  "memory_type_hint": "episodic | semantic | working | procedural | null"
}
```

**Response 201:**
```json
{
  "saved": [
    {
      "id": "mem_abc123",
      "memory_type": "episodic",
      "importance": 0.72,
      "token_count": 34,
      "created_at": "2024-01-20T10:30:00Z"
    }
  ],
  "discarded_count": 2
}
```

**Response 422:** Pydantic validation error (standard FastAPI format)

---

### GET /v1/memory/recall

Retrieve relevant memories for a query.

**Query params:**
| Param | Type | Default | Description |
|---|---|---|---|
| `user_id` | string | required | User identifier |
| `query` | string | required | Natural language query |
| `top_k` | int | 5 | Max memories to return |
| `token_budget` | int | 2000 | Max total tokens in response |
| `memory_types` | string | all | Comma-separated: `episodic,semantic,procedural,working` |

**Response 200:**
```json
{
  "memories": [
    {
      "id": "mem_abc123",
      "content": "User is a backend engineer...",
      "memory_type": "semantic",
      "importance": 0.89,
      "relevance_score": 0.91,
      "created_at": "2024-01-20T10:30:00Z",
      "session_id": "sess_xyz"
    }
  ],
  "total_tokens": 847,
  "budget_used": 0.42,
  "prompt_block": "<memory>\n[SEMANTIC] User is a backend engineer...\n</memory>"
}
```

---

### GET /v1/memory

List all memories for a user (paginated).

**Query params:**
| Param | Type | Default |
|---|---|---|
| `user_id` | string | required |
| `memory_type` | string | all |
| `page` | int | 1 |
| `page_size` | int | 20 |
| `include_compressed` | bool | false |

**Response 200:**
```json
{
  "items": [...],
  "total": 142,
  "page": 1,
  "page_size": 20
}
```

---

### DELETE /v1/memory/{memory_id}

Delete a specific memory.

**Response 200:**
```json
{ "deleted": true, "id": "mem_abc123" }
```

**Response 404:**
```json
{ "detail": "Memory not found" }
```

---

### DELETE /v1/memory

Delete all memories for a user (GDPR compliance).

**Request:**
```json
{ "user_id": "string (required)", "confirm": true }
```

**Response 200:**
```json
{ "deleted_count": 147 }
```

---

### GET /v1/session/{session_id}/stats

Get statistics for a session.

**Query params:**
| Param | Type | Default |
|---|---|---|
| `user_id` | string | required |

**Response 200:**
```json
{
  "session_id": "sess_xyz",
  "user_id": "user_123",
  "memory_count": 12,
  "total_tokens_stored": 3200,
  "started_at": "2024-01-20T10:00:00Z",
  "last_activity": "2024-01-20T11:30:00Z",
  "compressed": false
}
```

---

### POST /v1/session/{session_id}/compress

Trigger manual compression of a session.

**Query params:**
| Param | Type | Default |
|---|---|---|
| `user_id` | string | required |

**Response 202:**
```json
{
  "job_id": "job_abc123",
  "status": "queued",
  "message": "Compression queued. Check /v1/jobs/{job_id} for status."
}
```

---

### GET /v1/procedural

List procedural memory preferences for a user.

**Query params:**
| Param | Type | Default |
|---|---|---|
| `user_id` | string | required |

**Response 200:**
```json
{
  "items": [
    {
      "key": "tone",
      "value": "Use concise technical responses.",
      "confidence": 0.91,
      "updated_at": "2024-01-20T10:30:00Z",
      "source_chunk_id": "mem_abc123"
    }
  ]
}
```

---

### PUT /v1/procedural

Create or update one procedural memory preference for a user.

**Request:**
```json
{
  "user_id": "string (required)",
  "key": "string (required)",
  "value": "string (required)",
  "confidence": "float [0.0, 1.0] (optional, default: 1.0)",
  "source_chunk_id": "string | null"
}
```

**Response 200:**
```json
{
  "key": "tone",
  "value": "Use concise technical responses.",
  "confidence": 0.91,
  "updated_at": "2024-01-20T10:30:00Z",
  "source_chunk_id": "mem_abc123"
}
```

---

### DELETE /v1/procedural/{key}

Delete one procedural memory preference for a user.

**Query params:**
| Param | Type | Default |
|---|---|---|
| `user_id` | string | required |

**Response 200:**
```json
{
  "deleted": true,
  "key": "tone"
}
```

**Response 404:**
```json
{ "detail": "Procedural memory not found" }
```

---

### GET /v1/health

Health check endpoint.

**Response 200:**
```json
{
  "status": "ok",
  "version": "0.1.0",
  "storage": { "chroma": "ok", "sqlite": "ok" },
  "embedding_model": "all-MiniLM-L6-v2"
}
```

---

### GET /metrics

Prometheus exposition endpoint.

- Available only when `ML_METRICS_ENABLED=true`
- Not included in OpenAPI schema (`/openapi.json`)

**Response 200:** Prometheus text format (`text/plain; version=0.0.4`)

**Response 404:** when metrics are disabled

---

## Error Format

Authentication failures return:

```json
{
  "detail": "Unauthorized"
}
```

with response header `WWW-Authenticate: Bearer`.

All errors follow RFC 7807 Problem Details:

```json
{
  "type": "https://memory-vault-ai.dev/errors/invalid-user",
  "title": "Invalid user ID",
  "status": 422,
  "detail": "user_id must be a non-empty string",
  "instance": "/v1/memory"
}
```

## Rate Limits

Default enforced limits (configurable via env):
- `POST /v1/memory`: 100 req/min per user (`ML_RATE_LIMIT_SAVE`)
- `GET /v1/memory/recall`: 200 req/min per user (`ML_RATE_LIMIT_RECALL`)

When a limit is exceeded, the API returns:

```json
{
  "detail": "Rate limit exceeded"
}
```

with status `429 Too Many Requests` and header `Retry-After: <seconds>`.
