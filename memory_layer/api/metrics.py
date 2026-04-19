from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)


class ApiMetrics:
    """Prometheus metrics collector for API and memory operations."""

    def __init__(self) -> None:
        self._registry = CollectorRegistry(auto_describe=True)

        self._requests_total = Counter(
            "memory_layer_requests_total",
            "Total API requests by endpoint, method, and status.",
            labelnames=("endpoint", "method", "status"),
            registry=self._registry,
        )
        self._request_duration_seconds = Histogram(
            "memory_layer_request_duration_seconds",
            "Request duration in seconds by endpoint.",
            labelnames=("endpoint",),
            registry=self._registry,
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        )
        self._memories_total = Counter(
            "memory_layer_memories_total",
            "Total memories saved grouped by user and memory type.",
            labelnames=("user_id", "memory_type"),
            registry=self._registry,
        )
        self._recall_latency_seconds = Histogram(
            "memory_layer_recall_latency_seconds",
            "Latency in seconds for recall operations.",
            registry=self._registry,
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
        )
        self._ingestion_latency_seconds = Histogram(
            "memory_layer_ingestion_latency_seconds",
            "Latency in seconds for memory ingestion operations.",
            registry=self._registry,
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
        )
        self._token_budget_utilization = Histogram(
            "memory_layer_token_budget_utilization",
            "Token budget utilization ratio for recall operations.",
            registry=self._registry,
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        )

    @property
    def content_type(self) -> str:
        """Return the Prometheus content type for exposition output."""
        return CONTENT_TYPE_LATEST

    def observe_request(
        self,
        *,
        endpoint: str,
        method: str,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        """Record request throughput and duration metrics."""
        status_label = str(status_code)
        self._requests_total.labels(
            endpoint=endpoint,
            method=method,
            status=status_label,
        ).inc()
        self._request_duration_seconds.labels(endpoint=endpoint).observe(duration_seconds)

    def observe_ingestion_latency(self, duration_seconds: float) -> None:
        """Record latency for save/ingestion operations."""
        self._ingestion_latency_seconds.observe(duration_seconds)

    def observe_recall_latency(self, duration_seconds: float) -> None:
        """Record latency for recall operations."""
        self._recall_latency_seconds.observe(duration_seconds)

    def observe_token_budget_utilization(self, budget_used: float) -> None:
        """Record recall token budget utilization ratio."""
        normalized = min(max(budget_used, 0.0), 1.0)
        self._token_budget_utilization.observe(normalized)

    def increment_memories_total(self, *, user_id: str, memory_type: str, count: int = 1) -> None:
        """Increment the saved-memory counter for a user and memory type."""
        if count <= 0:
            return
        self._memories_total.labels(user_id=user_id, memory_type=memory_type).inc(count)

    def render_latest(self) -> bytes:
        """Render metrics in Prometheus text exposition format."""
        return generate_latest(self._registry)


__all__ = ["ApiMetrics"]
