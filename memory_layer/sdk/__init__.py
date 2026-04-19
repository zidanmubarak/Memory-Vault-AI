from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

from memory_layer.compression import CompressionResult, MemoryCompressor
from memory_layer.config import Settings, get_settings
from memory_layer.exceptions import ConfigurationError
from memory_layer.ingestion.chunker import SemanticChunker
from memory_layer.ingestion.embedder import SentenceTransformerEmbedder
from memory_layer.ingestion.engine import IngestionEngine
from memory_layer.ingestion.scorer import ImportanceScorer
from memory_layer.models import MemoryChunk, MemoryConfig, MemoryType, PaginatedResult, RecallResult
from memory_layer.retrieval.engine import RetrievalEngine
from memory_layer.storage import (
	ChromaAdapter,
	CompositeStorage,
	MemoryListQuery,
	ProceduralMemoryRecord,
	QdrantAdapter,
	SessionStatsRecord,
	SQLiteAdapter,
	StorageBackend,
)


def _session_id() -> str:
	"""Generate canonical session identifier."""
	return f"sess_{uuid4().hex[:8]}"


class IngestionPipeline(Protocol):
	"""Protocol for ingestion engine behavior used by SDK."""

	async def ingest(
		self,
		text: str,
		user_id: str,
		session_id: str,
		memory_type_hint: MemoryType | None = None,
	) -> list[MemoryChunk]:
		"""Ingest text for user/session and return saved chunks."""


class RetrievalPipeline(Protocol):
	"""Protocol for retrieval engine behavior used by SDK."""

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
		"""Recall relevant memories for a user query."""


class MemoryLayer:
	"""High-level public SDK interface for user-scoped memory operations."""

	def __init__(
		self,
		*,
		user_id: str,
		session_id: str | None = None,
		config: MemoryConfig | None = None,
		settings: Settings | None = None,
		storage: StorageBackend | None = None,
		ingestion_engine: IngestionPipeline | None = None,
		retrieval_engine: RetrievalPipeline | None = None,
	) -> None:
		if not user_id:
			raise ValueError("user_id is required")

		self.user_id = user_id
		self.session_id = session_id or _session_id()
		self.config = config or (settings or get_settings()).to_memory_config()

		self._storage = storage or self._build_storage(self.config)
		self._managed_embedders: list[SentenceTransformerEmbedder] = []

		ingestion_pipeline = ingestion_engine
		retrieval_pipeline = retrieval_engine

		if ingestion_pipeline is None and retrieval_pipeline is None:
			shared_embedder = self._new_embedder()
			self._managed_embedders.append(shared_embedder)
			ingestion_pipeline = self._new_ingestion_engine(shared_embedder)
			retrieval_pipeline = self._new_retrieval_engine(shared_embedder)
		elif ingestion_pipeline is None and retrieval_pipeline is not None:
			ingest_embedder = self._new_embedder()
			self._managed_embedders.append(ingest_embedder)
			ingestion_pipeline = self._new_ingestion_engine(ingest_embedder)
		elif ingestion_pipeline is not None and retrieval_pipeline is None:
			retrieval_embedder = self._new_embedder()
			self._managed_embedders.append(retrieval_embedder)
			retrieval_pipeline = self._new_retrieval_engine(retrieval_embedder)

		if ingestion_pipeline is None or retrieval_pipeline is None:
			raise ConfigurationError("SDK engines failed to initialize")

		self._ingestion_engine = ingestion_pipeline
		self._retrieval_engine = retrieval_pipeline

		self._initialized = False

	async def initialize(self) -> None:
		"""Initialize underlying storage backends."""
		if self._initialized:
			return
		await self._storage.initialize()
		self._initialized = True

	async def close(self) -> None:
		"""Close underlying resources and reset initialization state."""
		first_error: Exception | None = None

		for embedder in self._managed_embedders:
			try:
				await embedder.close()
			except Exception as exc:  # pragma: no cover - defensive cleanup
				if first_error is None:
					first_error = exc

		try:
			await self._storage.close()
		except Exception as exc:  # pragma: no cover - defensive cleanup
			if first_error is None:
				first_error = exc

		self._initialized = False

		if first_error is not None:
			raise first_error

	async def __aenter__(self) -> MemoryLayer:
		"""Initialize SDK resources when used as async context manager."""
		await self.initialize()
		return self

	async def __aexit__(
		self,
		exc_type: type[BaseException] | None,
		exc: BaseException | None,
		exc_tb: object,
	) -> None:
		"""End current session and close SDK resources on context exit."""
		del exc_type, exc, exc_tb
		try:
			await self.end_session()
		finally:
			await self.close()

	async def save(
		self,
		text: str,
		*,
		memory_type_hint: MemoryType | None = None,
		session_id: str | None = None,
	) -> list[MemoryChunk]:
		"""Save user text into memory chunks through ingestion pipeline."""
		await self.initialize()
		target_session = session_id or self.session_id
		return await self._ingestion_engine.ingest(
			text,
			self.user_id,
			target_session,
			memory_type_hint,
		)

	async def recall(
		self,
		query: str,
		*,
		top_k: int | None = None,
		token_budget: int | None = None,
		memory_types: list[MemoryType] | None = None,
		include_compressed: bool = False,
		reranker_enabled: bool | None = None,
	) -> RecallResult:
		"""Recall relevant memory chunks for a query."""
		await self.initialize()
		return await self._retrieval_engine.recall(
			query,
			self.user_id,
			top_k=top_k or self.config.top_k,
			token_budget=token_budget or self.config.token_budget,
			memory_types=memory_types,
			include_compressed=include_compressed,
			reranker_enabled=reranker_enabled,
		)

	async def list(
		self,
		*,
		memory_type: MemoryType | None = None,
		page: int = 1,
		page_size: int = 20,
		include_compressed: bool = False,
	) -> PaginatedResult[MemoryChunk]:
		"""List stored memories for current user with pagination."""
		await self.initialize()
		return await self._storage.list_memory_chunks(
			MemoryListQuery(
				user_id=self.user_id,
				memory_type=memory_type,
				include_compressed=include_compressed,
				page=page,
				page_size=page_size,
			)
		)

	async def upsert_procedural_memory(
		self,
		*,
		key: str,
		value: str,
		confidence: float = 1.0,
		source_chunk_id: str | None = None,
	) -> ProceduralMemoryRecord:
		"""Create or update one procedural memory preference for current user."""
		await self.initialize()
		record = ProceduralMemoryRecord(
			user_id=self.user_id,
			key=key,
			value=value,
			confidence=confidence,
			updated_at=datetime.now(UTC),
			source_chunk_id=source_chunk_id,
		)
		return await self._storage.upsert_procedural_memory(record)

	async def list_procedural_memory(self) -> Sequence[ProceduralMemoryRecord]:
		"""List procedural memory preferences for current user."""
		await self.initialize()
		return await self._storage.list_procedural_memory(user_id=self.user_id)

	async def delete_procedural_memory(self, *, key: str) -> bool:
		"""Delete one procedural memory preference for current user by key."""
		await self.initialize()
		return await self._storage.delete_procedural_memory(user_id=self.user_id, key=key)

	async def compress(
		self,
		*,
		force: bool = False,
		sessions_to_compress: int | None = None,
	) -> CompressionResult:
		"""Compress old episodic sessions for current user into summaries."""
		await self.initialize()
		compressor = MemoryCompressor(
			storage=self._storage,
			compression_threshold=self.config.compression_threshold,
		)
		return await compressor.compress_user(
			self.user_id,
			force=force,
			sessions_to_compress=sessions_to_compress,
		)

	async def forget(self, *, memory_id: str) -> bool:
		"""Delete one memory record by identifier."""
		await self.initialize()
		return await self._storage.delete_memory_chunk(
			memory_id=memory_id,
			user_id=self.user_id,
		)

	async def forget_all(self, *, confirm: bool = False) -> int:
		"""Delete all user memory data when explicitly confirmed."""
		if not confirm:
			raise ValueError("confirm must be true when forgetting all memories")

		await self.initialize()
		deleted = await self._storage.delete_memory_chunks_for_user(user_id=self.user_id)

		procedural = await self._storage.list_procedural_memory(user_id=self.user_id)
		for record in procedural:
			await self._storage.delete_procedural_memory(user_id=self.user_id, key=record.key)

		return deleted + len(procedural)

	async def end_session(self) -> SessionStatsRecord | None:
		"""Mark the current session as ended and persist final timestamp."""
		await self.initialize()

		current = await self._storage.get_session_stats(session_id=self.session_id)
		if current is None or current.user_id != self.user_id:
			return None

		now = datetime.now(UTC)
		last_activity = now if now >= current.started_at else current.started_at
		updated = SessionStatsRecord(
			session_id=current.session_id,
			user_id=current.user_id,
			memory_count=current.memory_count,
			total_tokens_stored=current.total_tokens_stored,
			started_at=current.started_at,
			last_activity=last_activity,
			ended_at=last_activity,
			compressed=current.compressed,
		)
		return await self._storage.upsert_session_stats(updated)

	@staticmethod
	def _build_storage(config: MemoryConfig) -> StorageBackend:
		"""Construct supported storage backend combination from SDK config."""
		vector_backend: ChromaAdapter | QdrantAdapter
		if config.storage_backend == "chroma":
			vector_backend = ChromaAdapter(chroma_path=config.chroma_path)
		elif config.storage_backend == "qdrant":
			if not config.qdrant_url:
				raise ConfigurationError(
					"qdrant_url is required when storage_backend='qdrant'"
				)
			vector_backend = QdrantAdapter(
				qdrant_url=config.qdrant_url,
				api_key=config.qdrant_api_key,
				collection_name=config.qdrant_collection,
			)
		else:
			raise ConfigurationError(
				"Unsupported storage backend. Use 'chroma' or 'qdrant'."
			)
		if config.metadata_backend != "sqlite":
			raise ConfigurationError(
				"Only metadata_backend='sqlite' is currently supported by the SDK"
			)

		metadata_backend = SQLiteAdapter(sqlite_path=config.sqlite_path)
		return CompositeStorage(
			vector_backend=vector_backend,
			metadata_backend=metadata_backend,
		)

	def _new_embedder(self) -> SentenceTransformerEmbedder:
		"""Build default embedder configured for this SDK instance."""
		return SentenceTransformerEmbedder(model_name=self.config.embedding_model)

	def _new_ingestion_engine(self, embedder: SentenceTransformerEmbedder) -> IngestionPipeline:
		"""Build default ingestion engine configured from SDK settings."""
		return IngestionEngine(
			storage=self._storage,
			chunker=SemanticChunker(
				min_chunk_tokens=self.config.min_chunk_tokens,
				max_chunk_tokens=self.config.max_chunk_tokens,
			),
			embedder=embedder,
			scorer=ImportanceScorer(threshold=self.config.importance_threshold),
		)

	def _new_retrieval_engine(self, embedder: SentenceTransformerEmbedder) -> RetrievalPipeline:
		"""Build default retrieval engine configured from SDK settings."""
		return RetrievalEngine(
			storage=self._storage,
			embedder=embedder,
			reranker_enabled=self.config.reranker_enabled,
			reranker_model=self.config.reranker_model,
		)

SDK_PUBLIC_METHODS: tuple[str, ...] = (
	"initialize",
	"close",
	"save",
	"recall",
	"list",
	"upsert_procedural_memory",
	"list_procedural_memory",
	"delete_procedural_memory",
	"compress",
	"forget",
	"forget_all",
	"end_session",
)

__all__ = ["SDK_PUBLIC_METHODS", "MemoryLayer"]
