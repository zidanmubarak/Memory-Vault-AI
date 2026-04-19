# Deployment Guide — Memory Layer AI

> **Audience:** DevOps engineers and developers deploying Memory Layer AI to production.
> For local development setup, see the [documentation home](../index.md).
> For SDK usage, see [`docs/guides/SDK_GUIDE.md`](SDK_GUIDE.md).

---

## Deployment Modes

| Mode | Use case | Storage | Scaling |
|---|---|---|---|
| **Embedded** | Single app, local use | ChromaDB on disk + SQLite | Single process |
| **API Server** | Multi-app, shared service | ChromaDB + SQLite (single node) | Vertical |
| **Production** | High traffic, multi-user | Qdrant cluster + PostgreSQL | Horizontal |
| **Docker Compose** | Self-hosted team | Qdrant + SQLite | Vertical |

---

## Option 1: Docker Compose (Recommended for Self-Hosting)

The repository ships with a production-oriented
[`Dockerfile`](https://github.com/zidanmubarak/Memory-Layer-AI/blob/main/Dockerfile)
and
[`docker-compose.yml`](https://github.com/zidanmubarak/Memory-Layer-AI/blob/main/docker-compose.yml).
The compose stack starts:

- `memory-layer-api` (FastAPI service)
- `qdrant` (vector store)
- persistent volumes for SQLite metadata and Qdrant storage

### Compose Configuration

The default compose stack uses these runtime settings:

- `ML_STORAGE_BACKEND=qdrant`
- `ML_QDRANT_URL=http://qdrant:6333`
- `ML_SQLITE_PATH=/app/data/memory.db`
- `ML_API_KEY=${ML_API_KEY:-}` (optional)

You can override values by creating a local `.env` file before startup.

### Start

```bash
# Optional: set API key used by Bearer auth middleware
echo "ML_API_KEY=replace-with-a-secret" >> .env

# Build and start all services
docker compose up --build -d

# Check service status
docker compose ps

# Verify
curl http://localhost:8000/v1/health
curl http://localhost:8000/openapi.json
```

---

## Option 2: Manual Server Deployment

### System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| CPU | 2 cores | 4 cores |
| RAM | 1 GB | 4 GB |
| Disk | 10 GB | 50 GB SSD |
| Python | 3.11 | 3.11+ |
| OS | Ubuntu 22.04 | Ubuntu 22.04+ |

### Install

```bash
# System dependencies
sudo apt-get update && sudo apt-get install -y python3.11 python3.11-venv

# Create service user
sudo useradd -m -s /bin/bash memorylayer

# Install as service user
sudo -u memorylayer bash -c "
  python3.11 -m venv /home/memorylayer/venv
  /home/memorylayer/venv/bin/pip install 'memory-layer-ai[qdrant]'
"
```

### Environment Configuration

Create `/etc/memory-layer/env`:

```bash
# Storage
ML_STORAGE_BACKEND=qdrant
ML_QDRANT_URL=http://localhost:6333
ML_SQLITE_PATH=/var/lib/memory-layer/memory.db

# Embedding
ML_EMBEDDING_MODEL=all-MiniLM-L6-v2

# Security
ML_API_KEY=your-secret-api-key-here

# Performance
ML_DEFAULT_TOKEN_BUDGET=2000
ML_DEFAULT_TOP_K=5
ML_RERANKER_ENABLED=false
ML_COMPRESSION_THRESHOLD=10

# Logging
ML_LOG_LEVEL=INFO
ML_LOG_FORMAT=json
```

### Systemd Service

Create `/etc/systemd/system/memory-layer-api.service`:

```ini
[Unit]
Description=Memory Layer AI API Server
After=network.target qdrant.service
Requires=qdrant.service

[Service]
Type=exec
User=memorylayer
Group=memorylayer
WorkingDirectory=/home/memorylayer
EnvironmentFile=/etc/memory-layer/env
ExecStart=/home/memorylayer/venv/bin/uvicorn \
    memory_layer.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --log-config /etc/memory-layer/log-config.json
ExecReload=/bin/kill -HUP $MAINPID
Restart=on-failure
RestartSec=5
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable memory-layer-api
sudo systemctl start memory-layer-api
sudo systemctl status memory-layer-api
```

---

## Option 3: Kubernetes

### Deployment manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: memory-layer-api
  namespace: memory-layer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: memory-layer-api
  template:
    metadata:
      labels:
        app: memory-layer-api
    spec:
      containers:
      - name: api
        image: ghcr.io/zidanmubarak/memory-layer-ai:latest
        ports:
        - containerPort: 8000
        env:
        - name: ML_STORAGE_BACKEND
          value: "qdrant"
        - name: ML_QDRANT_URL
          value: "http://qdrant-service:6333"
        - name: ML_API_KEY
          valueFrom:
            secretKeyRef:
              name: memory-layer-secrets
              key: api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /v1/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /v1/health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: memory-layer-service
  namespace: memory-layer
spec:
  selector:
    app: memory-layer-api
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

**Note:** When running multiple API replicas, all instances must share the same Qdrant
cluster and the same SQLite-compatible database (use PostgreSQL for multi-instance SQLite replacement — see ADR-007).

---

## Nginx Reverse Proxy

For production, put Memory Layer AI behind Nginx:

```nginx
upstream memory_layer {
    server 127.0.0.1:8000;
    keepalive 32;
}

server {
    listen 443 ssl http2;
    server_name memory.yourdomain.com;

    ssl_certificate     /etc/ssl/certs/memory.yourdomain.com.crt;
    ssl_certificate_key /etc/ssl/private/memory.yourdomain.com.key;

    location /v1/ {
        proxy_pass http://memory_layer;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts (compression jobs can take time)
        proxy_read_timeout 120s;
        proxy_connect_timeout 10s;
    }

    location /v1/health {
        proxy_pass http://memory_layer;
        access_log off;
    }
}

server {
    listen 80;
    server_name memory.yourdomain.com;
    return 301 https://$host$request_uri;
}
```

---

## Environment Variables — Full Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `ML_STORAGE_BACKEND` | No | `chroma` | `chroma` or `qdrant` |
| `ML_CHROMA_PATH` | No | `./data/chroma` | ChromaDB persistence directory |
| `ML_SQLITE_PATH` | No | `./data/memory.db` | SQLite database file |
| `ML_QDRANT_URL` | If Qdrant | — | e.g. `http://localhost:6333` |
| `ML_QDRANT_API_KEY` | No | — | Qdrant Cloud API key |
| `ML_QDRANT_COLLECTION` | No | `memory_layer` | Qdrant collection name |
| `ML_EMBEDDING_MODEL` | No | `all-MiniLM-L6-v2` | sentence-transformers model |
| `ML_API_KEY` | Prod | — | Enables Bearer auth on all endpoints |
| `ML_DEFAULT_TOKEN_BUDGET` | No | `2000` | Default token budget per recall |
| `ML_DEFAULT_TOP_K` | No | `5` | Default memories returned per recall |
| `ML_COMPRESSION_THRESHOLD` | No | `10` | Sessions before auto-compression |
| `ML_RERANKER_ENABLED` | No | `false` | Cross-encoder re-ranking (adds latency) |
| `ML_IMPORTANCE_THRESHOLD` | No | `0.3` | Min importance score to save a chunk |
| `ML_COMPRESSION_MODEL` | No | System default | LLM used for summarization |
| `ML_LOG_LEVEL` | No | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `ML_LOG_FORMAT` | No | `text` | `text` or `json` |
| `ML_WORKERS` | No | `1` | Uvicorn worker count |
| `ML_CORS_ORIGINS` | No | `*` | Comma-separated allowed origins |

---

## Observability

### Health endpoint

```bash
curl http://localhost:8000/v1/health
# {
#   "status": "ok",
#   "version": "0.1.0",
#   "storage": { "chroma": "ok", "sqlite": "ok" },
#   "embedding_model": "all-MiniLM-L6-v2",
#   "uptime_seconds": 3642
# }
```

### Prometheus metrics

Metrics exposed at `/metrics` (enable with `ML_METRICS_ENABLED=true`):

```
memory_layer_requests_total{endpoint, method, status}
memory_layer_request_duration_seconds{endpoint}
memory_layer_memories_total{user_id, memory_type}
memory_layer_recall_latency_seconds
memory_layer_ingestion_latency_seconds
memory_layer_token_budget_utilization
```

### Structured logging (JSON)

Set `ML_LOG_FORMAT=json` for log aggregation (Loki, CloudWatch, Datadog):

```json
{
  "timestamp": "2024-01-20T10:30:00Z",
  "level": "INFO",
  "event": "memory.saved",
  "user_id": "alice",
  "session_id": "sess_abc123",
  "memory_type": "semantic",
  "importance": 0.82,
  "chunks_saved": 2,
  "duration_ms": 45
}
```

---

## Data Backup

### ChromaDB backup

```bash
# Stop writes first (or use snapshot if available)
systemctl stop memory-layer-api

# Backup
tar -czf "chroma-backup-$(date +%Y%m%d).tar.gz" /var/lib/memory-layer/chroma/

# Restart
systemctl start memory-layer-api
```

### SQLite backup

```bash
# SQLite supports hot backup
sqlite3 /var/lib/memory-layer/memory.db ".backup /backups/memory-$(date +%Y%m%d).db"
```

### Automated backup script

```bash
#!/bin/bash
# /usr/local/bin/memory-layer-backup.sh
set -e

BACKUP_DIR="/backups/memory-layer"
DATE=$(date +%Y%m%d-%H%M%S)

mkdir -p "$BACKUP_DIR"

# SQLite hot backup
sqlite3 /var/lib/memory-layer/memory.db \
  ".backup $BACKUP_DIR/memory-$DATE.db"

# ChromaDB snapshot (stop API briefly)
systemctl stop memory-layer-api
tar -czf "$BACKUP_DIR/chroma-$DATE.tar.gz" /var/lib/memory-layer/chroma/
systemctl start memory-layer-api

# Retain 30 days
find "$BACKUP_DIR" -mtime +30 -delete

echo "Backup complete: $DATE"
```

Add to cron: `0 2 * * * /usr/local/bin/memory-layer-backup.sh`

---

## Scaling Considerations

| Bottleneck | Symptom | Solution |
|---|---|---|
| Embedding computation | High CPU, slow ingestion | Upgrade CPU; use GPU with `ML_EMBEDDING_DEVICE=cuda` |
| Vector search | Slow recall (>200ms) | Switch to Qdrant; enable HNSW index |
| SQLite write contention | 500 errors under concurrent writes | Migrate to PostgreSQL (see ADR-007) |
| Compression jobs blocking | API slowness during compression | Move compression to separate worker process |
| Memory growth | Disk full | Increase `ML_COMPRESSION_THRESHOLD`; add disk |

---

## Security Hardening Checklist

- [ ] `ML_API_KEY` set and rotated regularly
- [ ] Server bound to `localhost` only (Nginx handles external TLS)
- [ ] Qdrant not exposed on public network
- [ ] SQLite file permissions: `chmod 600 memory.db`
- [ ] Data directory permissions: `chmod 700 /var/lib/memory-layer`
- [ ] Backups encrypted at rest
- [ ] Log files do not contain memory content (set `ML_LOG_SANITIZE=true`)
- [ ] CORS origins explicitly set (not `*`) in production
