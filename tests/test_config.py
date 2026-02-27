"""Tests for configuration loading."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

from ai_search.config import AppConfig, load_config

if TYPE_CHECKING:
    from pathlib import Path


class TestAppConfig:
    """Test the AppConfig model and defaults."""

    def test_default_config(self) -> None:
        """Default config should have sensible values."""
        config = AppConfig()
        assert config.models.llm_model == "gpt-4o"
        assert config.models.embedding_model == "text-embedding-3-large"
        assert config.index.vector_dimensions.semantic == 3072
        assert config.index.vector_dimensions.structural == 1024
        assert config.index.vector_dimensions.style == 512

    def test_config_from_dict(self) -> None:
        """Config should load from a dict correctly."""
        data = {
            "models": {"llm_model": "custom-model"},
            "index": {"name": "my-index"},
        }
        config = AppConfig(**data)
        assert config.models.llm_model == "custom-model"
        assert config.index.name == "my-index"
        # Defaults preserved for unspecified fields
        assert config.models.embedding_model == "text-embedding-3-large"

    def test_search_weights_sum_to_one(self) -> None:
        """Default search weights should approximately sum to 1.0."""
        config = AppConfig()
        total = (
            config.search.semantic_weight
            + config.search.structural_weight
            + config.search.style_weight
            + config.search.image_weight
            + config.search.keyword_weight
        )
        assert abs(total - 1.0) < 1e-6


class TestLoadConfig:
    """Test YAML config loading."""

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        """Config should load from a YAML file."""
        config_data = {
            "models": {"llm_model": "yaml-model"},
            "index": {"name": "yaml-index"},
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        # Clear cache before test
        load_config.cache_clear()
        config = load_config(config_file)
        assert config.models.llm_model == "yaml-model"
        assert config.index.name == "yaml-index"
        load_config.cache_clear()

    def test_load_missing_file(self, tmp_path: Path) -> None:
        """Config should use defaults when file is missing."""
        load_config.cache_clear()
        config = load_config(tmp_path / "nonexistent.yaml")
        assert config.models.llm_model == "gpt-4o"
        load_config.cache_clear()
