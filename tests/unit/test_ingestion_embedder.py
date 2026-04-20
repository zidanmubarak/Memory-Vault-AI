from __future__ import annotations

import pytest

from memory_vault.exceptions import EmbeddingError
from memory_vault.ingestion.embedder import SentenceTransformerEmbedder


class FakeModel:
    def __init__(self) -> None:
        self.calls = 0

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> list[list[float]]:
        del batch_size, convert_to_numpy, normalize_embeddings
        self.calls += 1
        return [[float(len(text)), float(index)] for index, text in enumerate(texts)]


class BrokenModel:
    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> list[list[float]]:
        del texts, batch_size, convert_to_numpy, normalize_embeddings
        raise RuntimeError("encode failed")


@pytest.mark.asyncio
async def test_encode_batch_returns_embeddings_in_order() -> None:
    model = FakeModel()
    embedder = SentenceTransformerEmbedder(model=model, batch_size=8)

    vectors = await embedder.encode_batch(["alpha", "beta", "gamma"])

    assert vectors == [
        [5.0, 0.0],
        [4.0, 1.0],
        [5.0, 2.0],
    ]
    assert model.calls == 1


@pytest.mark.asyncio
async def test_encode_batch_uses_cache_for_duplicate_inputs() -> None:
    model = FakeModel()
    embedder = SentenceTransformerEmbedder(model=model, use_cache=True)

    first = await embedder.encode_batch(["same", "different"])
    second = await embedder.encode_batch(["same", "different"])

    assert first == second
    assert model.calls == 1
    assert embedder.cache_size == 2


@pytest.mark.asyncio
async def test_encode_batch_without_cache_calls_model_each_time() -> None:
    model = FakeModel()
    embedder = SentenceTransformerEmbedder(model=model, use_cache=False)

    await embedder.encode_batch(["same"])
    await embedder.encode_batch(["same"])

    assert model.calls == 2


@pytest.mark.asyncio
async def test_encode_batch_empty_input_returns_empty() -> None:
    embedder = SentenceTransformerEmbedder(model=FakeModel())

    vectors = await embedder.encode_batch([])

    assert vectors == []


@pytest.mark.asyncio
async def test_encode_batch_raises_embedding_error_on_model_failure() -> None:
    embedder = SentenceTransformerEmbedder(model=BrokenModel())

    with pytest.raises(EmbeddingError):
        await embedder.encode_batch(["alpha"])


def test_embedder_validates_batch_size() -> None:
    with pytest.raises(ValueError):
        SentenceTransformerEmbedder(model=FakeModel(), batch_size=0)
