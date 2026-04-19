from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import sqrt
from pathlib import Path
from random import Random
from time import perf_counter

from memory_layer.ingestion.engine import IngestionEngine
from memory_layer.ingestion.scorer import ImportanceScorer
from memory_layer.models import MemoryChunk, MemoryConfig, PaginatedResult
from memory_layer.retrieval.engine import RetrievalEngine
from memory_layer.sdk import MemoryLayer
from memory_layer.storage.base import (
    MemoryListQuery,
    MemorySearchQuery,
    ProceduralMemoryRecord,
    SessionStatsRecord,
    StorageBackend,
)


@dataclass(slots=True)
class BenchmarkConfig:
    save_count: int = 500
    recall_count: int = 300
    warmup_saves: int = 50
    warmup_recalls: int = 20
    top_k: int = 5
    token_budget: int = 512
    seed: int = 42


@dataclass(slots=True)
class LatencyStats:
    count: int
    min_ms: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float


@dataclass(slots=True)
class OperationStats:
    operation: str
    count: int
    duration_seconds: float
    throughput_ops_per_second: float
    latency_ms: LatencyStats


@dataclass(slots=True)
class BenchmarkReport:
    generated_at: str
    config: BenchmarkConfig
    operations: list[OperationStats]


class SingleChunker:
    def chunk(self, text: str) -> list[str]:
        return [text] if text.strip() else []


class DeterministicEmbedder:
    async def encode_batch(self, chunks: list[str]) -> list[list[float]]:
        return [self._encode(text) for text in chunks]

    @staticmethod
    def _encode(text: str) -> list[float]:
        lowered = text.lower()
        vector = [
            float(lowered.count("fastapi") + lowered.count("api") + lowered.count("endpoint")),
            float(lowered.count("postgres") + lowered.count("database") + lowered.count("sql")),
            float(
                lowered.count("prefer")
                + lowered.count("concise")
                + lowered.count("detailed")
                + lowered.count("format")
            ),
            float(lowered.count("cache") + lowered.count("redis") + lowered.count("memory")),
        ]
        if vector == [0.0, 0.0, 0.0, 0.0]:
            return [1.0, 0.0, 0.0, 0.0]
        return vector


class InMemoryStorage(StorageBackend):
    def __init__(self) -> None:
        self._initialized = False
        self._chunks: dict[str, MemoryChunk] = {}
        self._procedural: list[ProceduralMemoryRecord] = []
        self._sessions: dict[str, SessionStatsRecord] = {}

    async def initialize(self) -> None:
        self._initialized = True

    async def close(self) -> None:
        self._initialized = False

    async def healthcheck(self) -> dict[str, str]:
        return {
            "status": "ok",
            "vector": "ok",
            "metadata": "ok",
        }

    async def upsert_vectors(self, chunks: Sequence[MemoryChunk]) -> None:
        del chunks

    async def query_vectors(self, query: MemorySearchQuery) -> list[MemoryChunk]:
        candidates: list[MemoryChunk] = []

        for chunk in self._chunks.values():
            if chunk.user_id != query.user_id:
                continue
            if chunk.embedding is None:
                continue
            if chunk.importance < query.min_importance:
                continue
            if query.memory_types and chunk.memory_type not in query.memory_types:
                continue
            if chunk.compressed and not query.include_compressed:
                continue

            score = self._cosine_similarity(query.query_embedding, chunk.embedding)
            candidates.append(chunk.model_copy(update={"relevance_score": score}))

        candidates.sort(key=lambda chunk: chunk.relevance_score or 0.0, reverse=True)
        return candidates[: query.top_k]

    async def delete_vectors(self, memory_ids: Sequence[str], *, user_id: str) -> int:
        deleted = 0
        for memory_id in memory_ids:
            chunk = self._chunks.get(memory_id)
            if chunk is not None and chunk.user_id == user_id:
                self._chunks.pop(memory_id)
                deleted += 1
        return deleted

    async def delete_vectors_for_user(self, *, user_id: str) -> int:
        target_ids = [
            memory_id
            for memory_id, chunk in self._chunks.items()
            if chunk.user_id == user_id
        ]
        for memory_id in target_ids:
            self._chunks.pop(memory_id)
        return len(target_ids)

    async def upsert_memory_chunks(self, chunks: Sequence[MemoryChunk]) -> list[MemoryChunk]:
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
        return list(chunks)

    async def get_memory_chunk(self, *, memory_id: str, user_id: str) -> MemoryChunk | None:
        chunk = self._chunks.get(memory_id)
        if chunk is None or chunk.user_id != user_id:
            return None
        return chunk

    async def list_memory_chunks(self, query: MemoryListQuery) -> PaginatedResult[MemoryChunk]:
        items = [
            chunk
            for chunk in self._chunks.values()
            if chunk.user_id == query.user_id
            and (query.memory_type is None or chunk.memory_type is query.memory_type)
            and (query.include_compressed or not chunk.compressed)
        ]
        items.sort(key=lambda chunk: chunk.created_at, reverse=True)
        total = len(items)
        start = (query.page - 1) * query.page_size
        end = start + query.page_size
        return PaginatedResult[MemoryChunk](
            items=items[start:end],
            total=total,
            page=query.page,
            page_size=query.page_size,
        )

    async def delete_memory_chunk(self, *, memory_id: str, user_id: str) -> bool:
        chunk = self._chunks.get(memory_id)
        if chunk is None or chunk.user_id != user_id:
            return False
        self._chunks.pop(memory_id)
        return True

    async def delete_memory_chunks_for_user(self, *, user_id: str) -> int:
        memory_ids = [
            memory_id
            for memory_id, chunk in self._chunks.items()
            if chunk.user_id == user_id
        ]
        for memory_id in memory_ids:
            self._chunks.pop(memory_id)
        return len(memory_ids)

    async def upsert_procedural_memory(
        self,
        record: ProceduralMemoryRecord,
    ) -> ProceduralMemoryRecord:
        filtered = [
            existing
            for existing in self._procedural
            if not (existing.user_id == record.user_id and existing.key == record.key)
        ]
        filtered.append(record)
        self._procedural = filtered
        return record

    async def list_procedural_memory(self, *, user_id: str) -> list[ProceduralMemoryRecord]:
        return [record for record in self._procedural if record.user_id == user_id]

    async def delete_procedural_memory(self, *, user_id: str, key: str) -> bool:
        initial = len(self._procedural)
        self._procedural = [
            record
            for record in self._procedural
            if not (record.user_id == user_id and record.key == key)
        ]
        return len(self._procedural) < initial

    async def upsert_session_stats(self, record: SessionStatsRecord) -> SessionStatsRecord:
        self._sessions[record.session_id] = record
        return record

    async def get_session_stats(self, *, session_id: str) -> SessionStatsRecord | None:
        return self._sessions.get(session_id)

    @staticmethod
    def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        if len(a) != len(b):
            return 0.0

        dot = sum(left * right for left, right in zip(a, b, strict=True))
        norm_a = sqrt(sum(value * value for value in a))
        norm_b = sqrt(sum(value * value for value in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0

    index = (len(sorted_values) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _latency_stats(latencies_ms: Sequence[float]) -> LatencyStats:
    sorted_values = sorted(latencies_ms)
    mean_ms = (sum(sorted_values) / len(sorted_values)) if sorted_values else 0.0
    return LatencyStats(
        count=len(sorted_values),
        min_ms=(sorted_values[0] if sorted_values else 0.0),
        mean_ms=mean_ms,
        p50_ms=_percentile(sorted_values, 0.50),
        p95_ms=_percentile(sorted_values, 0.95),
        p99_ms=_percentile(sorted_values, 0.99),
        max_ms=(sorted_values[-1] if sorted_values else 0.0),
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _build_messages(*, count: int, seed: int) -> list[str]:
    random = Random(seed)
    stacks = [
        "FastAPI with PostgreSQL",
        "Django with MySQL",
        "Flask with SQLite",
        "Express with Redis",
    ]
    preferences = [
        "I prefer concise answers.",
        "I prefer detailed explanations.",
        "Use bullet points in responses.",
        "Include concrete code snippets.",
    ]

    messages: list[str] = []
    for index in range(count):
        stack = random.choice(stacks)
        preference = random.choice(preferences)
        messages.append(
            f"Message {index}: I am building APIs using {stack}. {preference} "
            f"Current milestone index is {index}."
        )
    return messages


def _build_queries(*, count: int, seed: int) -> list[str]:
    random = Random(seed + 7)
    prompts = [
        "What stack is the user using?",
        "What response style does the user prefer?",
        "Summarize the latest project details.",
        "Which database is mentioned most often?",
    ]
    return [random.choice(prompts) for _ in range(count)]


async def _run_save_benchmark(
    *,
    memory: MemoryLayer,
    messages: Sequence[str],
) -> OperationStats:
    latencies_ms: list[float] = []
    started = perf_counter()

    for index, message in enumerate(messages):
        session_id = f"sess_bench_{index % 8}"
        op_started = perf_counter()
        await memory.save(message, session_id=session_id)
        latencies_ms.append((perf_counter() - op_started) * 1000.0)

    duration_seconds = perf_counter() - started
    throughput = len(messages) / duration_seconds if duration_seconds > 0 else 0.0
    return OperationStats(
        operation="save",
        count=len(messages),
        duration_seconds=duration_seconds,
        throughput_ops_per_second=throughput,
        latency_ms=_latency_stats(latencies_ms),
    )


async def _run_recall_benchmark(
    *,
    memory: MemoryLayer,
    queries: Sequence[str],
    top_k: int,
    token_budget: int,
) -> OperationStats:
    latencies_ms: list[float] = []
    started = perf_counter()

    for query in queries:
        op_started = perf_counter()
        await memory.recall(query, top_k=top_k, token_budget=token_budget)
        latencies_ms.append((perf_counter() - op_started) * 1000.0)

    duration_seconds = perf_counter() - started
    throughput = len(queries) / duration_seconds if duration_seconds > 0 else 0.0
    return OperationStats(
        operation="recall",
        count=len(queries),
        duration_seconds=duration_seconds,
        throughput_ops_per_second=throughput,
        latency_ms=_latency_stats(latencies_ms),
    )


async def run_benchmark(config: BenchmarkConfig) -> BenchmarkReport:
    storage = InMemoryStorage()
    embedder = DeterministicEmbedder()
    ingestion = IngestionEngine(
        storage=storage,
        chunker=SingleChunker(),
        embedder=embedder,
        scorer=ImportanceScorer(threshold=0.0),
        token_counter=lambda text: len(text.split()),
    )
    retrieval = RetrievalEngine(
        storage=storage,
        embedder=embedder,
        reranker_enabled=False,
        token_counter=lambda text: len(text.split()),
    )
    memory = MemoryLayer(
        user_id="benchmark_user",
        session_id="sess_benchmark",
        config=MemoryConfig(
            top_k=config.top_k,
            token_budget=config.token_budget,
            importance_threshold=0.0,
            reranker_enabled=False,
        ),
        storage=storage,
        ingestion_engine=ingestion,
        retrieval_engine=retrieval,
    )

    try:
        warmup_messages = _build_messages(count=config.warmup_saves, seed=config.seed)
        for index, message in enumerate(warmup_messages):
            await memory.save(message, session_id=f"warmup_{index % 4}")

        warmup_queries = _build_queries(count=config.warmup_recalls, seed=config.seed)
        for query in warmup_queries:
            await memory.recall(query, top_k=config.top_k, token_budget=config.token_budget)

        save_messages = _build_messages(count=config.save_count, seed=config.seed + 100)
        recall_queries = _build_queries(count=config.recall_count, seed=config.seed + 100)

        save_stats = await _run_save_benchmark(memory=memory, messages=save_messages)
        recall_stats = await _run_recall_benchmark(
            memory=memory,
            queries=recall_queries,
            top_k=config.top_k,
            token_budget=config.token_budget,
        )

        return BenchmarkReport(
            generated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            config=config,
            operations=[save_stats, recall_stats],
        )
    finally:
        await memory.close()


def _format_text_report(report: BenchmarkReport) -> str:
    lines = [
        "Memory Layer AI Benchmark Report",
        f"Generated: {report.generated_at}",
        (
            "Config: "
            f"save_count={report.config.save_count}, "
            f"recall_count={report.config.recall_count}, "
            f"top_k={report.config.top_k}, "
            f"token_budget={report.config.token_budget}"
        ),
        "",
    ]

    for operation in report.operations:
        latency = operation.latency_ms
        lines.extend(
            [
                f"[{operation.operation}]",
                f"  Count: {operation.count}",
                f"  Duration: {operation.duration_seconds:.4f}s",
                f"  Throughput: {operation.throughput_ops_per_second:.2f} ops/s",
                (
                    "  Latency (ms): "
                    f"min={latency.min_ms:.3f}, mean={latency.mean_ms:.3f}, "
                    f"p50={latency.p50_ms:.3f}, p95={latency.p95_ms:.3f}, "
                    f"p99={latency.p99_ms:.3f}, max={latency.max_ms:.3f}"
                ),
                "",
            ]
        )

    return "\n".join(lines).rstrip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Memory Layer AI performance benchmarks.")
    parser.add_argument("--save-count", type=_positive_int, default=500)
    parser.add_argument("--recall-count", type=_positive_int, default=300)
    parser.add_argument("--warmup-saves", type=_positive_int, default=50)
    parser.add_argument("--warmup-recalls", type=_positive_int, default=20)
    parser.add_argument("--top-k", type=_positive_int, default=5)
    parser.add_argument("--token-budget", type=_positive_int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional path to write report output.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = BenchmarkConfig(
        save_count=args.save_count,
        recall_count=args.recall_count,
        warmup_saves=args.warmup_saves,
        warmup_recalls=args.warmup_recalls,
        top_k=args.top_k,
        token_budget=args.token_budget,
        seed=args.seed,
    )

    report = asyncio.run(run_benchmark(config))

    payload_dict = {
        "generated_at": report.generated_at,
        "config": asdict(report.config),
        "operations": [
            {
                "operation": operation.operation,
                "count": operation.count,
                "duration_seconds": operation.duration_seconds,
                "throughput_ops_per_second": operation.throughput_ops_per_second,
                "latency_ms": asdict(operation.latency_ms),
            }
            for operation in report.operations
        ],
    }
    output = (
        json.dumps(payload_dict, indent=2)
        if args.format == "json"
        else _format_text_report(report)
    )

    if args.output_file is not None:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(output + "\n", encoding="utf-8")

    print(output)


if __name__ == "__main__":
    main()
