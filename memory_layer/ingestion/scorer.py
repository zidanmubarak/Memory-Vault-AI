from __future__ import annotations

import math
import re


def _clamp(value: float, *, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a float into the inclusive [low, high] interval."""
    if value < low:
        return low
    if value > high:
        return high
    return value


def _dot(left: list[float], right: list[float]) -> float:
    """Return vector dot product."""
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    )


def _norm(vector: list[float]) -> float:
    """Return euclidean norm of a vector."""
    return math.sqrt(sum(component * component for component in vector))


class ImportanceScorer:
    """Compute chunk importance from novelty and salience heuristics."""

    _PREFERENCE_SIGNAL_RE = re.compile(
        r"\b(i\s+prefer|i\s+always|i\s+hate|my\s+favorite)\b",
        re.IGNORECASE,
    )
    _NUMBER_OR_DATE_RE = re.compile(r"\b\d{1,4}([/-]\d{1,2}([/-]\d{1,4})?)?\b")
    _PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
    _TECH_TERM_RE = re.compile(
        r"\b(api|async|postgresql|docker|kubernetes|vector|embedding|token|schema)\b",
        re.IGNORECASE,
    )

    def __init__(self, *, threshold: float = 0.3) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        self.threshold = threshold

    def score(
        self,
        chunk_text: str,
        *,
        chunk_embedding: list[float],
        existing_embeddings: list[list[float]] | None = None,
    ) -> float:
        """Return weighted importance score in [0.0, 1.0]."""
        novelty_score = self.novelty(chunk_embedding, existing_embeddings or [])
        salience_score = self.salience(chunk_text)
        combined = (novelty_score * 0.6) + (salience_score * 0.4)
        return _clamp(combined)

    def is_important(
        self,
        chunk_text: str,
        *,
        chunk_embedding: list[float],
        existing_embeddings: list[list[float]] | None = None,
    ) -> bool:
        """Return whether score is above configured persistence threshold."""
        return self.score(
            chunk_text,
            chunk_embedding=chunk_embedding,
            existing_embeddings=existing_embeddings,
        ) >= self.threshold

    def novelty(
        self,
        chunk_embedding: list[float],
        existing_embeddings: list[list[float]],
    ) -> float:
        """Compute novelty from cosine distance to user memory centroid."""
        if not chunk_embedding:
            return 0.0
        if not existing_embeddings:
            return 1.0

        centroid = self._centroid(existing_embeddings)
        if not centroid:
            return 1.0

        similarity = self._cosine_similarity(chunk_embedding, centroid)
        distance = 1.0 - similarity
        return _clamp(distance)

    def salience(self, chunk_text: str) -> float:
        """Compute heuristic salience score from text signals."""
        text = chunk_text.strip()
        if not text:
            return 0.0

        score = 0.0

        if self._PROPER_NOUN_RE.search(text):
            score += 0.3
        if self._NUMBER_OR_DATE_RE.search(text):
            score += 0.2
        if self._PREFERENCE_SIGNAL_RE.search(text):
            score += 0.4
        if self._TECH_TERM_RE.search(text):
            score += 0.1

        return _clamp(score)

    @staticmethod
    def _centroid(embeddings: list[list[float]]) -> list[float]:
        """Return centroid vector for non-empty equally-shaped embeddings."""
        if not embeddings:
            return []

        dim = len(embeddings[0])
        if dim == 0:
            return []

        if any(len(vector) != dim for vector in embeddings):
            raise ValueError("existing_embeddings must have equal vector dimensions")

        sums = [0.0] * dim
        for vector in embeddings:
            for index, value in enumerate(vector):
                sums[index] += value

        count = float(len(embeddings))
        return [value / count for value in sums]

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        """Return cosine similarity for equally-sized vectors."""
        if len(left) != len(right):
            raise ValueError("embedding vectors must have matching dimensions")
        if not left:
            return 0.0

        left_norm = _norm(left)
        right_norm = _norm(right)
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0

        similarity = _dot(left, right) / (left_norm * right_norm)
        return _clamp(similarity, low=-1.0, high=1.0)


__all__ = ["ImportanceScorer"]
