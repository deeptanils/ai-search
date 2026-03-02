"""Tests for the index schema definition."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ai_search.indexing.schema import build_index_schema


class TestBuildIndexSchema:
    """Test the index schema builder."""

    @patch("ai_search.indexing.schema.load_config")
    def test_field_count(self, mock_config: MagicMock) -> None:
        """Schema should contain the expected number of fields."""
        from ai_search.config import AppConfig

        mock_config.return_value = AppConfig()

        index = build_index_schema()

        # 20 primitive fields + 4 primary vectors = 24
        assert len(index.fields) == 24

    @patch("ai_search.indexing.schema.load_config")
    def test_key_field(self, mock_config: MagicMock) -> None:
        """image_id should be the key field."""
        from ai_search.config import AppConfig

        mock_config.return_value = AppConfig()

        index = build_index_schema()
        key_fields = [f for f in index.fields if getattr(f, "key", False)]
        assert len(key_fields) == 1
        assert key_fields[0].name == "image_id"

    @patch("ai_search.indexing.schema.load_config")
    def test_vector_field_dimensions(self, mock_config: MagicMock) -> None:
        """Vector fields should have correct dimensions."""
        from ai_search.config import AppConfig

        mock_config.return_value = AppConfig()

        index = build_index_schema()
        vector_fields = {
            f.name: f.vector_search_dimensions
            for f in index.fields
            if hasattr(f, "vector_search_dimensions") and f.vector_search_dimensions
        }

        assert vector_fields["semantic_vector"] == 3072
        assert vector_fields["structural_vector"] == 1024
        assert vector_fields["style_vector"] == 512

    @patch("ai_search.indexing.schema.load_config")
    def test_hnsw_profile(self, mock_config: MagicMock) -> None:
        """Schema should have HNSW vector search configuration."""
        from ai_search.config import AppConfig

        mock_config.return_value = AppConfig()

        index = build_index_schema()
        assert index.vector_search is not None
        assert len(index.vector_search.algorithms) == 1
        assert index.vector_search.algorithms[0].name == "hnsw-cosine"
