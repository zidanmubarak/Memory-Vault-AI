from __future__ import annotations

from memory_layer.exceptions import BudgetExceededError, MemoryLayerError, StorageError


def test_budget_exceeded_error_exposes_budget_fields() -> None:
    error = BudgetExceededError(provided_budget=25, minimum_required=80)

    assert error.provided_budget == 25
    assert error.minimum_required == 80
    assert error.code == "budget_exceeded"
    assert "minimum required is 80" in str(error)


def test_domain_exceptions_share_common_base_type() -> None:
    error = StorageError("storage unavailable")

    assert isinstance(error, MemoryLayerError)
    assert error.code is None
