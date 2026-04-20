from __future__ import annotations


class MemoryLayerError(Exception):
    """Base class for all Memory Vault domain errors."""

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


class ConfigurationError(MemoryLayerError):
    """Raised when runtime configuration is invalid."""


class StorageError(MemoryLayerError):
    """Raised when vector or metadata storage operations fail."""


class IngestionError(MemoryLayerError):
    """Raised when ingestion pipeline execution fails."""


class EmbeddingError(IngestionError):
    """Raised when embedding generation fails."""


class RetrievalError(MemoryLayerError):
    """Raised when retrieval pipeline execution fails."""


class CompressionError(MemoryLayerError):
    """Raised when memory compression jobs fail."""


class NotFoundError(MemoryLayerError):
    """Raised when a requested domain entity is missing."""


class UserNotFoundError(NotFoundError):
    """Raised when no records exist for a user."""


class MemoryNotFoundError(NotFoundError):
    """Raised when a memory record cannot be found."""


class BudgetExceededError(RetrievalError):
    """Raised when token budget is too low to include any memory."""

    def __init__(self, provided_budget: int, minimum_required: int) -> None:
        self.provided_budget = provided_budget
        self.minimum_required = minimum_required
        message = (
            f"Token budget {provided_budget} is too small; "
            f"minimum required is {minimum_required}."
        )
        super().__init__(message, code="budget_exceeded")
