from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest

from memory_vault.models import MemoryChunk, MemoryType
from memory_vault.retrieval.reranker import MemoryReranker

_BASE_TIME = datetime(2026, 4, 19, 12, 0, tzinfo=UTC)


def _chunk(
    *,
    memory_id: str,
    relevance_score: float | None,
    importance: float,
    updated_at: datetime,
) -> MemoryChunk:
    return MemoryChunk(
        id=memory_id,
        user_id="user_a",
        session_id="sess_a",
        memory_type=MemoryType.SEMANTIC,
        content=f"content-{memory_id}",
        importance=importance,
        token_count=4,
        embedding=[0.1, 0.2],
        relevance_score=relevance_score,
        created_at=_BASE_TIME,
        updated_at=updated_at,
    )


def test_reranker_validates_weight_inputs() -> None:
    with pytest.raises(ValueError):
        MemoryReranker(relevance_weight=-0.1, importance_weight=0.2)
    with pytest.raises(ValueError):
        MemoryReranker(relevance_weight=0.2, importance_weight=-0.1)
    with pytest.raises(ValueError):
        MemoryReranker(relevance_weight=0.0, importance_weight=0.0)


def test_rerank_returns_empty_for_empty_candidates() -> None:
    reranker = MemoryReranker()

    assert reranker.rerank([]) == []


def test_rerank_replaces_scores_with_weighted_blend() -> None:
    reranker = MemoryReranker(relevance_weight=0.8, importance_weight=0.2)
    original = _chunk(
        memory_id="mem_a",
        relevance_score=0.5,
        importance=0.9,
        updated_at=_BASE_TIME,
    )

    ranked = reranker.rerank([original])

    assert ranked[0].relevance_score == pytest.approx(0.58)
    assert original.relevance_score == 0.5


def test_rerank_treats_missing_similarity_as_zero() -> None:
    reranker = MemoryReranker(relevance_weight=0.7, importance_weight=0.3)

    ranked = reranker.rerank(
        [
            _chunk(
                memory_id="mem_a",
                relevance_score=None,
                importance=0.9,
                updated_at=_BASE_TIME,
            )
        ]
    )

    assert ranked[0].relevance_score == pytest.approx(0.27)


def test_rerank_applies_top_k_and_recency_tie_break() -> None:
    reranker = MemoryReranker(relevance_weight=1.0, importance_weight=0.0)

    older = _chunk(
        memory_id="mem_old",
        relevance_score=0.8,
        importance=0.3,
        updated_at=_BASE_TIME,
    )
    newer = _chunk(
        memory_id="mem_new",
        relevance_score=0.8,
        importance=0.3,
        updated_at=_BASE_TIME + timedelta(minutes=1),
    )

    ranked = reranker.rerank([older, newer], top_k=1)

    assert [chunk.id for chunk in ranked] == ["mem_new"]


def test_rerank_validates_top_k() -> None:
    reranker = MemoryReranker()
    candidate = _chunk(
        memory_id="mem_a",
        relevance_score=0.3,
        importance=0.6,
        updated_at=_BASE_TIME,
    )

    with pytest.raises(ValueError):
        reranker.rerank([candidate], top_k=0)


class _StubCrossEncoderScorer:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.calls: list[list[tuple[str, str]]] = []

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        self.calls.append(list(pairs))
        return self._scores


class _FailingCrossEncoderScorer:
    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        del pairs
        raise RuntimeError("inference failed")


def test_rerank_uses_cross_encoder_when_query_present() -> None:
    scorer = _StubCrossEncoderScorer(scores=[2.0, -2.0])
    reranker = MemoryReranker(cross_encoder_scorer=scorer)
    first = _chunk(
        memory_id="mem_a",
        relevance_score=0.1,
        importance=0.1,
        updated_at=_BASE_TIME,
    )
    second = _chunk(
        memory_id="mem_b",
        relevance_score=0.9,
        importance=0.9,
        updated_at=_BASE_TIME + timedelta(minutes=1),
    )

    ranked = reranker.rerank([first, second], query_text="memory query")

    assert scorer.calls == [[("memory query", "content-mem_a"), ("memory query", "content-mem_b")]]
    assert [chunk.id for chunk in ranked] == ["mem_a", "mem_b"]
    assert ranked[0].relevance_score == pytest.approx(0.880797, abs=1e-5)
    assert ranked[1].relevance_score == pytest.approx(0.119203, abs=1e-5)


def test_rerank_falls_back_to_weighted_scores_when_cross_encoder_fails() -> None:
    reranker = MemoryReranker(
        relevance_weight=0.8,
        importance_weight=0.2,
        cross_encoder_scorer=_FailingCrossEncoderScorer(),
    )

    ranked = reranker.rerank(
        [
            _chunk(
                memory_id="mem_a",
                relevance_score=0.5,
                importance=0.9,
                updated_at=_BASE_TIME,
            )
        ],
        query_text="memory query",
    )

    assert ranked[0].relevance_score == pytest.approx(0.58)
