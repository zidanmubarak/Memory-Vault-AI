from __future__ import annotations

import pytest

from memory_layer.ingestion.scorer import ImportanceScorer


def test_scorer_validates_threshold_bounds() -> None:
    with pytest.raises(ValueError):
        ImportanceScorer(threshold=-0.1)
    with pytest.raises(ValueError):
        ImportanceScorer(threshold=1.1)


def test_novelty_is_one_when_user_has_no_existing_memory() -> None:
    scorer = ImportanceScorer()

    novelty = scorer.novelty([1.0, 0.0], [])

    assert novelty == 1.0


def test_novelty_uses_cosine_distance_to_centroid() -> None:
    scorer = ImportanceScorer()

    novelty = scorer.novelty(
        [1.0, 0.0],
        [
            [1.0, 0.0],
            [1.0, 0.0],
        ],
    )

    assert novelty == 0.0


def test_salience_combines_expected_signals() -> None:
    scorer = ImportanceScorer()

    text = "I prefer PostgreSQL in 2026 at Acme because async APIs are reliable."

    salience = scorer.salience(text)

    assert salience == 1.0


def test_score_applies_weighted_formula() -> None:
    scorer = ImportanceScorer()

    score = scorer.score(
        "I prefer APIs.",
        chunk_embedding=[1.0, 0.0],
        existing_embeddings=[[1.0, 0.0]],
    )

    # novelty=0.0, salience=0.4 => 0.0*0.6 + 0.4*0.4 = 0.16
    assert score == pytest.approx(0.16)


def test_is_important_respects_threshold() -> None:
    scorer = ImportanceScorer(threshold=0.3)

    is_important = scorer.is_important(
        "Acme shipped API v2 in 2026.",
        chunk_embedding=[0.0, 1.0],
        existing_embeddings=[[1.0, 0.0]],
    )

    assert is_important is True


def test_score_raises_for_dimension_mismatch() -> None:
    scorer = ImportanceScorer()

    with pytest.raises(ValueError):
        scorer.score(
            "text",
            chunk_embedding=[1.0, 0.0],
            existing_embeddings=[[1.0]],
        )
