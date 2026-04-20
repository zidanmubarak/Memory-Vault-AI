from __future__ import annotations

import json
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from threading import Lock
from time import monotonic
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, Response

from memory_vault import __version__
from memory_vault.api.metrics import ApiMetrics
from memory_vault.api.routes import memory_router, procedural_router, session_router
from memory_vault.api.ui_page import MEMORY_INTROSPECTION_HTML
from memory_vault.config import Settings, get_settings
from memory_vault.exceptions import ConfigurationError
from memory_vault.models import MemoryConfig
from memory_vault.storage import (
    ChromaAdapter,
    CompositeStorage,
    QdrantAdapter,
    SQLiteAdapter,
    StorageBackend,
)


def _build_storage(config: MemoryConfig) -> StorageBackend:
    """Build storage backend graph from runtime memory config."""
    vector_backend: ChromaAdapter | QdrantAdapter
    if config.storage_backend == "chroma":
        vector_backend = ChromaAdapter(chroma_path=config.chroma_path)
    elif config.storage_backend == "qdrant":
        if not config.qdrant_url:
            raise ConfigurationError("qdrant_url is required when storage_backend='qdrant'")
        vector_backend = QdrantAdapter(
            qdrant_url=config.qdrant_url,
            api_key=config.qdrant_api_key,
            collection_name=config.qdrant_collection,
        )
    else:
        raise ConfigurationError("Unsupported storage backend. Use 'chroma' or 'qdrant'.")
    if config.metadata_backend != "sqlite":
        raise ConfigurationError(
            "Only metadata_backend='sqlite' is currently supported by the API"
        )

    metadata_backend = SQLiteAdapter(sqlite_path=config.sqlite_path)
    return CompositeStorage(
        vector_backend=vector_backend,
        metadata_backend=metadata_backend,
    )


def _map_storage_health(
    settings: Settings,
    health_payload: dict[str, str],
) -> dict[str, str]:
    """Map backend health payload into API response storage schema."""
    if "vector" in health_payload or "metadata" in health_payload:
        return {
            settings.storage_backend: health_payload.get("vector", "unknown"),
            settings.metadata_backend: health_payload.get("metadata", "unknown"),
        }

    backend = health_payload.get("backend")
    status = health_payload.get("status", "unknown")
    if backend:
        return {backend: status}

    return {
        settings.storage_backend: status,
        settings.metadata_backend: status,
    }


def _parse_bearer_token(authorization: str | None) -> str | None:
    """Extract bearer token from Authorization header when present."""
    if not authorization:
        return None

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return None

    normalized = token.strip()
    if not normalized:
        return None
    return normalized


def _unauthorized_response() -> JSONResponse:
    """Build a standard unauthorized response payload."""
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={"detail": "Unauthorized"},
        headers={"WWW-Authenticate": "Bearer"},
    )


def _rate_limited_response(*, retry_after: int) -> JSONResponse:
    """Build a standard rate-limited response payload."""
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": "Rate limit exceeded"},
        headers={"Retry-After": str(retry_after)},
    )


def _rate_limit_for_request(request: Request, settings: Settings) -> int | None:
    """Return configured per-minute limit for supported request routes."""
    method = request.method.upper()
    path = request.url.path

    if method == "POST" and path == "/v1/memory":
        return settings.rate_limit_save
    if method == "GET" and path == "/v1/memory/recall":
        return settings.rate_limit_recall
    return None


def _metrics_endpoint_label(request: Request) -> str:
    """Return a stable endpoint label for request-level metrics."""
    route = request.scope.get("route")
    path_template = getattr(route, "path", None)
    if isinstance(path_template, str) and path_template:
        return path_template
    return request.url.path


async def _extract_user_id_for_rate_limit(request: Request) -> str | None:
    """Extract user identifier from query/body for rate-limited routes."""
    query_user = request.query_params.get("user_id")
    if query_user:
        return query_user

    method = request.method.upper()
    path = request.url.path
    if method == "POST" and path == "/v1/memory":
        raw_body = await request.body()
        if not raw_body:
            return None

        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            return None

        if not isinstance(payload, dict):
            return None

        user_id = payload.get("user_id")
        if isinstance(user_id, str) and user_id:
            return user_id

    return None


class _InMemoryRateLimiter:
    """Thread-safe in-memory sliding-window rate limiter."""

    def __init__(self, *, window_seconds: int = 60) -> None:
        self._window_seconds = window_seconds
        self._buckets: dict[str, deque[float]] = {}
        self._lock = Lock()

    def allow(self, *, key: str, limit: int) -> tuple[bool, int]:
        """Return whether a request should pass and retry-after when rejected."""
        now = monotonic()
        window_start = now - self._window_seconds

        with self._lock:
            bucket = self._buckets.setdefault(key, deque())

            while bucket and bucket[0] <= window_start:
                bucket.popleft()

            if len(bucket) >= limit:
                retry_after = max(1, int(self._window_seconds - (now - bucket[0])) + 1)
                return False, retry_after

            bucket.append(now)
            return True, 0


def create_app(
    *,
    settings: Settings | None = None,
    storage: StorageBackend | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    resolved_settings = settings or get_settings()
    resolved_storage = storage or _build_storage(resolved_settings.to_memory_config())
    rate_limiter = _InMemoryRateLimiter(window_seconds=60)
    api_metrics = ApiMetrics() if resolved_settings.metrics_enabled else None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = resolved_settings
        app.state.storage = resolved_storage
        app.state.metrics = api_metrics
        await resolved_storage.initialize()
        try:
            yield
        finally:
            await resolved_storage.close()

    app = FastAPI(
        title="Memory Vault AI",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    @app.middleware("http")
    async def security_and_rate_limit_middleware(request: Request, call_next: Any) -> Any:
        """Apply auth and per-user rate limits for configured API routes."""
        start_time = monotonic()
        path = request.url.path
        try:
            response: Any
            if path.startswith("/v1/"):
                response = None
                if resolved_settings.api_key and path != "/v1/health":
                    token = _parse_bearer_token(request.headers.get("Authorization"))
                    if token != resolved_settings.api_key:
                        response = _unauthorized_response()

                if response is None:
                    route_limit = _rate_limit_for_request(request, resolved_settings)
                    if route_limit is not None:
                        user_id = await _extract_user_id_for_rate_limit(request)
                        if user_id:
                            allowed, retry_after = rate_limiter.allow(
                                key=f"{request.method.upper()}:{path}:{user_id}",
                                limit=route_limit,
                            )
                            if not allowed:
                                response = _rate_limited_response(retry_after=retry_after)

                if response is None:
                    response = await call_next(request)
            else:
                response = await call_next(request)
        except Exception:
            if api_metrics is not None:
                api_metrics.observe_request(
                    endpoint=_metrics_endpoint_label(request),
                    method=request.method.upper(),
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    duration_seconds=monotonic() - start_time,
                )
            raise

        if api_metrics is not None:
            api_metrics.observe_request(
                endpoint=_metrics_endpoint_label(request),
                method=request.method.upper(),
                status_code=response.status_code,
                duration_seconds=monotonic() - start_time,
            )
        return response

    app.include_router(memory_router)
    app.include_router(procedural_router)
    app.include_router(session_router)

    @app.get("/ui", include_in_schema=False)
    async def memory_introspection_ui() -> HTMLResponse:
        """Serve browser-based memory introspection UI."""
        return HTMLResponse(content=MEMORY_INTROSPECTION_HTML)

    if api_metrics is not None:

        @app.get("/metrics", include_in_schema=False)
        async def metrics() -> Response:
            """Expose Prometheus metrics when enabled by configuration."""
            return Response(
                content=api_metrics.render_latest(),
                media_type=api_metrics.content_type,
            )

    @app.get("/v1/health")
    async def health() -> dict[str, Any]:
        storage_health = await resolved_storage.healthcheck()
        return {
            "status": storage_health.get("status", "unknown"),
            "version": __version__,
            "storage": _map_storage_health(resolved_settings, storage_health),
            "embedding_model": resolved_settings.embedding_model,
        }

    return app


app = create_app()


__all__ = ["app", "create_app"]
