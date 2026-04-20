from __future__ import annotations

import importlib
import re
import unicodedata
from collections.abc import Callable


def _whitespace_token_count(text: str) -> int:
    """Return a fallback token estimate based on whitespace splitting."""
    return len(text.split())


def _build_tiktoken_counter(encoding_name: str) -> Callable[[str], int] | None:
    """Build a tiktoken-based token counter when dependency is available."""
    try:
        tiktoken = importlib.import_module("tiktoken")
    except ModuleNotFoundError:
        return None

    encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(text: str) -> int:
        return len(encoding.encode(text))

    return count_tokens


class SemanticChunker:
    """Split cleaned text into semantically coherent chunks with token bounds."""

    _SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
    _NEWLINE_COLLAPSE_RE = re.compile(r"\n{3,}")

    def __init__(
        self,
        *,
        min_chunk_tokens: int = 50,
        max_chunk_tokens: int = 300,
        encoding_name: str = "cl100k_base",
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        if min_chunk_tokens <= 0:
            raise ValueError("min_chunk_tokens must be greater than zero")
        if max_chunk_tokens <= 0:
            raise ValueError("max_chunk_tokens must be greater than zero")
        if min_chunk_tokens > max_chunk_tokens:
            raise ValueError("min_chunk_tokens must be less than or equal to max_chunk_tokens")

        self.min_chunk_tokens = min_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens
        self._token_count = (
            token_counter
            or _build_tiktoken_counter(encoding_name)
            or _whitespace_token_count
        )

    def clean_text(self, text: str) -> str:
        """Normalize text while preserving formatting markers and code blocks."""
        normalized = unicodedata.normalize("NFC", text)
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = normalized.strip()
        if not normalized:
            return ""
        return self._NEWLINE_COLLAPSE_RE.sub("\n\n", normalized)

    def chunk(self, text: str) -> list[str]:
        """Chunk text using semantic boundaries and token-size constraints."""
        cleaned = self.clean_text(text)
        if not cleaned:
            return []

        primary_segments = [segment.strip() for segment in cleaned.split("\n\n") if segment.strip()]

        expanded: list[str] = []
        for segment in primary_segments:
            if self._token_count(segment) <= self.max_chunk_tokens:
                expanded.append(segment)
            else:
                expanded.extend(self._split_long_segment(segment))

        merged = self._merge_short_chunks(expanded)

        final_chunks: list[str] = []
        for segment in merged:
            if self._token_count(segment) <= self.max_chunk_tokens:
                final_chunks.append(segment)
            else:
                final_chunks.extend(self._split_long_segment(segment))

        return [chunk for chunk in final_chunks if chunk.strip()]

    def _split_long_segment(self, segment: str) -> list[str]:
        """Split oversized segments at sentence boundaries with whitespace fallback."""
        sentences = [
            part.strip()
            for part in self._SENTENCE_BOUNDARY_RE.split(segment)
            if part.strip()
        ]
        if len(sentences) <= 1:
            return self._split_by_token_limit(segment)

        chunks: list[str] = []
        current_parts: list[str] = []

        for sentence in sentences:
            if self._token_count(sentence) > self.max_chunk_tokens:
                if current_parts:
                    chunks.append(" ".join(current_parts).strip())
                    current_parts = []
                chunks.extend(self._split_by_token_limit(sentence))
                continue

            candidate_parts = [*current_parts, sentence]
            candidate = " ".join(candidate_parts).strip()

            if self._token_count(candidate) <= self.max_chunk_tokens:
                current_parts = candidate_parts
            else:
                if current_parts:
                    chunks.append(" ".join(current_parts).strip())
                current_parts = [sentence]

        if current_parts:
            chunks.append(" ".join(current_parts).strip())

        return [chunk for chunk in chunks if chunk]

    def _split_by_token_limit(self, text: str) -> list[str]:
        """Fallback splitter for oversized text with no sentence boundaries."""
        words = text.split()
        if not words:
            return []

        chunks: list[str] = []
        current_words: list[str] = []

        for word in words:
            candidate_words = [*current_words, word]
            candidate = " ".join(candidate_words)
            if current_words and self._token_count(candidate) > self.max_chunk_tokens:
                chunks.append(" ".join(current_words))
                current_words = [word]
            else:
                current_words = candidate_words

        if current_words:
            chunks.append(" ".join(current_words))

        return chunks

    def _merge_short_chunks(self, chunks: list[str]) -> list[str]:
        """Merge chunks smaller than min bound with adjacent chunks."""
        if not chunks:
            return []

        merged: list[str] = []
        pending = chunks[:]
        index = 0

        while index < len(pending):
            chunk = pending[index]
            token_count = self._token_count(chunk)

            if token_count >= self.min_chunk_tokens or len(pending) == 1:
                merged.append(chunk)
                index += 1
                continue

            if merged:
                candidate = f"{merged[-1]}\n\n{chunk}".strip()
                if self._token_count(candidate) <= self.max_chunk_tokens:
                    merged[-1] = candidate
                    index += 1
                    continue

            if index + 1 < len(pending):
                pending[index + 1] = f"{chunk}\n\n{pending[index + 1]}".strip()
                index += 1
                continue

            merged.append(chunk)
            index += 1

        return merged


__all__ = ["SemanticChunker"]
