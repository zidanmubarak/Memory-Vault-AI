from __future__ import annotations

import pytest
from pydantic import ValidationError

from memory_vault.config import Settings, get_settings


def test_settings_maps_to_memory_config() -> None:
    settings = Settings()

    config = settings.to_memory_config()

    assert config.token_budget == settings.default_token_budget
    assert config.top_k == settings.default_top_k
    assert config.embedding_model == settings.embedding_model
    assert config.reranker_model == settings.reranker_model


def test_settings_validates_qdrant_requirements() -> None:
    with pytest.raises(ValidationError):
        Settings(storage_backend="qdrant", qdrant_url=None)


def test_settings_validates_postgres_requirements() -> None:
    with pytest.raises(ValidationError):
        Settings(metadata_backend="postgres", postgres_url=None)


def test_settings_reads_env_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ML_DEFAULT_TOP_K", "9")

    settings = Settings()

    assert settings.default_top_k == 9


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()

    first = get_settings()
    second = get_settings()

    assert first is second
