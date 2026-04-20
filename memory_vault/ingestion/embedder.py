from __future__ import annotations

import hashlib
import importlib
from typing import Any

import anyio

from memory_vault.exceptions import ConfigurationError, EmbeddingError


class SentenceTransformerEmbedder:
    """Async sentence-transformers wrapper with optional in-memory caching."""

    def __init__(
        self,
        *,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
        use_cache: bool = True,
        model: Any | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.use_cache = use_cache

        self._model = model
        self._owns_model = model is None
        self._initialized = model is not None
        self._cache: dict[str, list[float]] = {}

    async def initialize(self) -> None:
        """Initialize and load the sentence-transformers model."""
        if self._initialized:
            return

        if self._model is None:
            try:
                sentence_transformers = importlib.import_module("sentence_transformers")
            except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency path
                raise ConfigurationError(
                    "SentenceTransformerEmbedder requires sentence-transformers. "
                    "Install memory-vault-ai[embeddings]."
                ) from exc

            try:
                self._model = await anyio.to_thread.run_sync(
                    lambda: sentence_transformers.SentenceTransformer(
                        self.model_name,
                        device=self.device,
                    )
                )
            except Exception as exc:  # pragma: no cover - defensive wrapper
                raise EmbeddingError(f"Failed to initialize embedding model: {exc}") from exc

        self._initialized = True

    async def close(self) -> None:
        """Release model references and reset initialization state."""
        if self._owns_model:
            self._model = None
        self._initialized = False

    async def encode_batch(self, chunks: list[str]) -> list[list[float]]:
        """Encode chunk list into vector embeddings, preserving input order."""
        if not chunks:
            return []

        await self.initialize()
        model = self._model
        if model is None:
            raise EmbeddingError("Embedding model is not initialized")

        embeddings: list[list[float] | None] = [None] * len(chunks)
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        if self.use_cache:
            for index, chunk in enumerate(chunks):
                cache_key = self._cache_key(chunk)
                cached = self._cache.get(cache_key)
                if cached is not None:
                    embeddings[index] = cached
                else:
                    missing_indices.append(index)
                    missing_texts.append(chunk)
        else:
            missing_indices = list(range(len(chunks)))
            missing_texts = chunks

        if missing_texts:
            try:
                new_vectors = await anyio.to_thread.run_sync(
                    lambda: model.encode(
                        missing_texts,
                        batch_size=self.batch_size,
                        convert_to_numpy=False,
                        normalize_embeddings=False,
                    )
                )
            except Exception as exc:
                raise EmbeddingError(f"Failed to encode embeddings: {exc}") from exc

            for idx, vector in enumerate(new_vectors):
                target_index = missing_indices[idx]
                converted = self._to_float_list(vector)
                embeddings[target_index] = converted
                if self.use_cache:
                    self._cache[self._cache_key(chunks[target_index])] = converted

        final_embeddings = [vector for vector in embeddings if vector is not None]
        if len(final_embeddings) != len(chunks):
            raise EmbeddingError("Embedding generation returned an unexpected vector count")

        return final_embeddings

    def clear_cache(self) -> None:
        """Clear all in-memory embedding cache entries."""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        """Return number of cached embeddings currently stored."""
        return len(self._cache)

    @staticmethod
    def _cache_key(text: str) -> str:
        """Build deterministic cache key from chunk content."""
        return hashlib.sha256(text.encode()).hexdigest()

    @staticmethod
    def _to_float_list(vector: Any) -> list[float]:
        """Convert encoder output vector to a Python float list."""
        if hasattr(vector, "tolist"):
            raw = vector.tolist()
        else:
            raw = list(vector)
        return [float(value) for value in raw]


__all__ = ["SentenceTransformerEmbedder"]
