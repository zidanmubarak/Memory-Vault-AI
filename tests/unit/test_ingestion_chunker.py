from __future__ import annotations

import pytest

from memory_vault.ingestion.chunker import SemanticChunker


def _token_count(text: str) -> int:
    return len(text.split())


def test_chunker_validates_token_bounds() -> None:
    with pytest.raises(ValueError):
        SemanticChunker(min_chunk_tokens=0)
    with pytest.raises(ValueError):
        SemanticChunker(max_chunk_tokens=0)
    with pytest.raises(ValueError):
        SemanticChunker(min_chunk_tokens=10, max_chunk_tokens=5)


def test_clean_text_normalizes_and_collapses_newlines() -> None:
    chunker = SemanticChunker(min_chunk_tokens=1, max_chunk_tokens=10, token_counter=_token_count)

    raw = "  Cafe\u0301\r\n\r\n\r\nLine two  "

    assert chunker.clean_text(raw) == "Café\n\nLine two"


def test_chunk_returns_empty_for_blank_input() -> None:
    chunker = SemanticChunker(min_chunk_tokens=1, max_chunk_tokens=10, token_counter=_token_count)

    assert chunker.chunk("   \n\n  ") == []


def test_chunk_splits_long_paragraph_at_sentence_boundaries() -> None:
    chunker = SemanticChunker(min_chunk_tokens=1, max_chunk_tokens=4, token_counter=_token_count)

    text = "One two three four. Five six seven eight. Nine ten eleven twelve."

    chunks = chunker.chunk(text)

    assert chunks == [
        "One two three four.",
        "Five six seven eight.",
        "Nine ten eleven twelve.",
    ]


def test_chunk_merges_short_adjacent_segments() -> None:
    chunker = SemanticChunker(min_chunk_tokens=3, max_chunk_tokens=8, token_counter=_token_count)

    text = "one\n\ntwo three four five"

    chunks = chunker.chunk(text)

    assert chunks == ["one\n\ntwo three four five"]


def test_chunk_falls_back_to_whitespace_split_for_long_sentence() -> None:
    chunker = SemanticChunker(min_chunk_tokens=1, max_chunk_tokens=5, token_counter=_token_count)

    text = "w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12"

    chunks = chunker.chunk(text)

    assert chunks == [
        "w1 w2 w3 w4 w5",
        "w6 w7 w8 w9 w10",
        "w11 w12",
    ]
