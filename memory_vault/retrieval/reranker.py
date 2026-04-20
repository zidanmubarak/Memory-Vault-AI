from __future__ import annotations

import importlib
from collections.abc import Sequence
from math import exp
from typing import Any, Protocol

from memory_vault.models import MemoryChunk


def _clamp(value: float, *, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a float into the inclusive [low, high] interval."""
    if value < low:
        return low
    if value > high:
        return high
    return value


class MemoryReranker:
    """Reranker with optional cross-encoder scoring and weighted fallback."""

    def __init__(
        self,
        *,
        relevance_weight: float = 0.8,
        importance_weight: float = 0.2,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        cross_encoder_scorer: CrossEncoderScorer | None = None,
    ) -> None:
        if relevance_weight < 0.0:
            raise ValueError("relevance_weight must be non-negative")
        if importance_weight < 0.0:
            raise ValueError("importance_weight must be non-negative")

        total_weight = relevance_weight + importance_weight
        if total_weight == 0.0:
            raise ValueError("At least one reranker weight must be greater than zero")

        self._relevance_weight = relevance_weight / total_weight
        self._importance_weight = importance_weight / total_weight
        self._cross_encoder_model = cross_encoder_model
        self._cross_encoder_scorer = cross_encoder_scorer

    def rerank(
        self,
        candidates: Sequence[MemoryChunk],
        *,
        top_k: int | None = None,
        query_text: str | None = None,
    ) -> list[MemoryChunk]:
        """Return candidates ordered by blended relevance and optionally truncated."""
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be greater than zero when provided")
        if not candidates:
            return []

        cross_encoder_scores = self._score_with_cross_encoder(
            query_text=query_text,
            candidates=candidates,
        )

        rescored: list[MemoryChunk]
        if cross_encoder_scores is None:
            rescored = [self._score_chunk(chunk) for chunk in candidates]
        else:
            rescored = [
                chunk.model_copy(update={"relevance_score": score})
                for chunk, score in zip(candidates, cross_encoder_scores, strict=True)
            ]
        rescored.sort(
            key=lambda chunk: (chunk.relevance_score or 0.0, chunk.updated_at),
            reverse=True,
        )

        if top_k is None:
            return rescored
        return rescored[:top_k]

    def _score_chunk(self, chunk: MemoryChunk) -> MemoryChunk:
        """Compute weighted score and return a copied chunk with updated relevance."""
        similarity = chunk.relevance_score or 0.0
        blended = (similarity * self._relevance_weight) + (
            chunk.importance * self._importance_weight
        )
        return chunk.model_copy(update={"relevance_score": _clamp(blended)})

    def _score_with_cross_encoder(
        self,
        *,
        query_text: str | None,
        candidates: Sequence[MemoryChunk],
    ) -> list[float] | None:
        """Return normalized cross-encoder scores, or None when unavailable."""
        if query_text is None or not query_text.strip():
            return None

        scorer = self._cross_encoder_scorer
        if scorer is None:
            scorer = self._load_default_cross_encoder()
            if scorer is None:
                return None
            self._cross_encoder_scorer = scorer

        pairs = [(query_text, chunk.content) for chunk in candidates]
        try:
            raw_scores = scorer.score_pairs(pairs)
        except Exception:
            return None

        try:
            scores = [self._normalize_score(score) for score in raw_scores]
        except Exception:
            return None
        if len(scores) != len(candidates):
            return None
        return scores

    def _load_default_cross_encoder(self) -> CrossEncoderScorer | None:
        """Lazily load sentence-transformers CrossEncoder scorer when available."""
        try:
            sentence_transformers = importlib.import_module("sentence_transformers")
            cross_encoder_cls = sentence_transformers.CrossEncoder
            model = cross_encoder_cls(self._cross_encoder_model)
        except Exception:
            return None
        return _ModelCrossEncoderScorer(model)

    @staticmethod
    def _normalize_score(raw_score: float) -> float:
        """Normalize raw cross-encoder score into [0.0, 1.0] interval."""
        score = float(raw_score)
        if score >= 0.0:
            z = exp(-score)
            return _clamp(1.0 / (1.0 + z))
        z = exp(score)
        return _clamp(z / (1.0 + z))


class CrossEncoderScorer(Protocol):
    """Protocol for query-document scoring using a cross-encoder model."""

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        """Return one relevance score per input query-content pair."""


class _ModelCrossEncoderScorer:
    """Cross-encoder scorer wrapper around sentence-transformers CrossEncoder."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        raw = self._model.predict(list(pairs), convert_to_numpy=False)
        if hasattr(raw, "tolist"):
            return [float(value) for value in raw.tolist()]
        return [float(value) for value in raw]


__all__ = ["CrossEncoderScorer", "MemoryReranker"]
