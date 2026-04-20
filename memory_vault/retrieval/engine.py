from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

from memory_vault.budget.manager import ContextBudgetManager
from memory_vault.exceptions import BudgetExceededError, MemoryLayerError, RetrievalError
from memory_vault.models import MemoryChunk, MemoryType, RecallResult
from memory_vault.prompt.builder import PromptBuilder
from memory_vault.retrieval.reranker import MemoryReranker
from memory_vault.retrieval.searcher import MemorySearcher
from memory_vault.storage.base import ProceduralMemoryRecord, StorageBackend


class QueryEmbedder(Protocol):
    """Protocol for query embedding components."""

    async def encode_batch(self, chunks: list[str]) -> list[list[float]]:
        """Encode chunk list into vector embeddings."""


class CandidateSearcher(Protocol):
    """Protocol for memory candidate search components."""

    async def search(
        self,
        *,
        user_id: str,
        query_embedding: Sequence[float],
        top_k: int = 5,
        memory_types: Sequence[MemoryType] | None = None,
        include_compressed: bool = False,
    ) -> list[MemoryChunk]:
        """Search memory candidates for a query embedding."""


class CandidateReranker(Protocol):
    """Protocol for reranking candidate memories."""

    def rerank(
        self,
        candidates: Sequence[MemoryChunk],
        *,
        top_k: int | None = None,
        query_text: str | None = None,
    ) -> list[MemoryChunk]:
        """Return reranked candidates."""


class BudgetManager(Protocol):
    """Protocol for token budget selection components."""

    def select(
        self,
        *,
        procedural_memories: Sequence[MemoryChunk],
        ranked_memories: Sequence[MemoryChunk],
        top_k: int,
        token_budget: int,
    ) -> tuple[list[MemoryChunk], int]:
        """Select memories within token and top-k limits."""

    def minimum_tokens(self, memories: Sequence[MemoryChunk]) -> int:
        """Return the minimum token count across provided memory candidates."""

    def count_tokens(self, text: str) -> int:
        """Count tokens for a text payload."""


class MemoryPromptBuilder(Protocol):
    """Protocol for prompt assembly components."""

    def build(self, memories: Sequence[MemoryChunk]) -> str:
        """Build the prompt block from selected memories."""


class RetrievalEngine:
    """Retrieval orchestrator for query embedding, search, reranking, and budgeting."""

    def __init__(
        self,
        *,
        storage: StorageBackend,
        embedder: QueryEmbedder,
        searcher: CandidateSearcher | None = None,
        reranker: CandidateReranker | None = None,
        budget_manager: BudgetManager | None = None,
        prompt_builder: MemoryPromptBuilder | None = None,
        reranker_enabled: bool = False,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        candidate_pool_multiplier: int = 4,
        token_counter: Callable[[str], int] | None = None,
        encoding_name: str = "cl100k_base",
    ) -> None:
        if candidate_pool_multiplier <= 0:
            raise ValueError("candidate_pool_multiplier must be greater than zero")

        self._storage = storage
        self._embedder = embedder
        self._searcher = searcher or MemorySearcher(storage=storage)
        self._reranker = reranker or MemoryReranker(cross_encoder_model=reranker_model)
        self._budget_manager = budget_manager or ContextBudgetManager(
            token_counter=token_counter,
            encoding_name=encoding_name,
        )
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._reranker_enabled = reranker_enabled
        self._candidate_pool_multiplier = candidate_pool_multiplier

    async def recall(
        self,
        query: str,
        user_id: str,
        *,
        top_k: int = 5,
        token_budget: int = 2000,
        memory_types: Sequence[MemoryType] | None = None,
        include_compressed: bool = False,
        reranker_enabled: bool | None = None,
    ) -> RecallResult:
        """Recall relevant memories for a user query within a token budget."""
        if not query.strip():
            raise ValueError("query is required")
        if not user_id:
            raise ValueError("user_id is required")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if token_budget <= 0:
            raise ValueError("token_budget must be greater than zero")

        normalized_types = tuple(memory_types) if memory_types else None
        include_procedural = (
            normalized_types is None or MemoryType.PROCEDURAL in normalized_types
        )
        search_types: tuple[MemoryType, ...] | None
        if normalized_types is None:
            search_types = None
        else:
            search_types = tuple(
                memory_type
                for memory_type in normalized_types
                if memory_type is not MemoryType.PROCEDURAL
            )

        use_reranker = self._reranker_enabled if reranker_enabled is None else reranker_enabled
        search_top_k = max(top_k, top_k * self._candidate_pool_multiplier)

        try:
            query_vectors = await self._embedder.encode_batch([query])
            if not query_vectors:
                raise RetrievalError("Query embedding generation returned no vectors")

            query_embedding = query_vectors[0]

            procedural_memories = await self._load_procedural_memories(user_id=user_id)
            if not include_procedural:
                procedural_memories = []

            if search_types == ():
                candidates: list[MemoryChunk] = []
            else:
                candidates = await self._searcher.search(
                    user_id=user_id,
                    query_embedding=query_embedding,
                    top_k=search_top_k,
                    memory_types=search_types,
                    include_compressed=include_compressed,
                )

            ranked = (
                self._reranker.rerank(
                    candidates,
                    top_k=search_top_k,
                    query_text=query,
                )
                if use_reranker
                else candidates
            )

            selected, tokens_used = self._budget_manager.select(
                procedural_memories=procedural_memories,
                ranked_memories=ranked,
                top_k=top_k,
                token_budget=token_budget,
            )

            if not selected and (procedural_memories or ranked):
                min_tokens = self._budget_manager.minimum_tokens(
                    [*procedural_memories, *ranked]
                )
                raise BudgetExceededError(token_budget, min_tokens)

            budget_used = tokens_used / token_budget if token_budget else 0.0
            prompt_block = self._prompt_builder.build(selected)
            return RecallResult(
                memories=selected,
                total_tokens=tokens_used,
                budget_used=budget_used,
                prompt_block=prompt_block,
            )
        except MemoryLayerError:
            raise
        except Exception as exc:
            raise RetrievalError(f"Failed to recall memories: {exc}") from exc

    async def _load_procedural_memories(self, *, user_id: str) -> list[MemoryChunk]:
        """Load procedural key-value records and map them to MemoryChunk objects."""
        records = await self._storage.list_procedural_memory(user_id=user_id)
        mapped = [self._procedural_record_to_chunk(record) for record in records]
        mapped.sort(key=lambda chunk: chunk.updated_at, reverse=True)
        return mapped

    def _procedural_record_to_chunk(self, record: ProceduralMemoryRecord) -> MemoryChunk:
        """Map one procedural record to memory-chunk shape used by recall results."""
        token_count = self._budget_manager.count_tokens(record.value)
        return MemoryChunk(
            id=f"proc_{record.key}",
            user_id=record.user_id,
            session_id="procedural",
            memory_type=MemoryType.PROCEDURAL,
            content=record.value,
            importance=record.confidence,
            token_count=token_count,
            embedding=None,
            created_at=record.updated_at,
            updated_at=record.updated_at,
            metadata={
                "procedural_key": record.key,
                "source_chunk_id": record.source_chunk_id,
            },
        )


__all__ = ["RetrievalEngine"]
