"""Tests for multi-vector and hybrid search execution."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ai_search.retrieval.search import execute_hybrid_search, execute_vector_search


class TestExecuteVectorSearch:
    """Test the pure multi-vector search function."""

    @patch("ai_search.retrieval.search.get_search_client")
    @patch("ai_search.retrieval.search.load_config")
    def test_builds_vector_queries_no_bm25(
        self,
        mock_config: MagicMock,
        mock_client_factory: MagicMock,
    ) -> None:
        """Should build VectorizedQuery objects without BM25 text search."""
        from ai_search.config import AppConfig

        mock_config.return_value = AppConfig()

        mock_client = MagicMock()
        mock_client_factory.return_value = mock_client

        mock_result = {
            "image_id": "test-1",
            "generation_prompt": "test",
            "@search.score": 0.95,
        }
        mock_client.search.return_value = [mock_result]

        query_vectors = {
            "semantic_vector": [0.1] * 3072,
            "structural_vector": [0.2] * 1024,
            "style_vector": [0.3] * 512,
        }

        results = execute_vector_search(query_vectors)

        assert len(results) == 1
        assert results[0]["search_score"] == 1.0  # normalized: single result → 1.0
        mock_client.search.assert_called_once()

        call_kwargs = mock_client.search.call_args
        assert call_kwargs.kwargs["search_text"] is None
        assert len(call_kwargs.kwargs["vector_queries"]) == 3

    @patch("ai_search.retrieval.search.get_search_client")
    @patch("ai_search.retrieval.search.load_config")
    def test_includes_image_vector_for_image_mode(
        self,
        mock_config: MagicMock,
        mock_client_factory: MagicMock,
    ) -> None:
        """Should include image_vector query when present (image search mode)."""
        from ai_search.config import AppConfig

        mock_config.return_value = AppConfig()

        mock_client = MagicMock()
        mock_client_factory.return_value = mock_client
        mock_client.search.return_value = []

        query_vectors = {
            "semantic_vector": [0.1] * 3072,
            "structural_vector": [0.2] * 1024,
            "style_vector": [0.3] * 512,
            "image_vector": [0.4] * 1024,
        }

        execute_vector_search(query_vectors)

        call_kwargs = mock_client.search.call_args
        assert len(call_kwargs.kwargs["vector_queries"]) == 4

    @patch("ai_search.retrieval.search.get_search_client")
    @patch("ai_search.retrieval.search.load_config")
    def test_passes_odata_filter(
        self,
        mock_config: MagicMock,
        mock_client_factory: MagicMock,
    ) -> None:
        """Should pass OData filter to Azure search."""
        from ai_search.config import AppConfig

        mock_config.return_value = AppConfig()

        mock_client = MagicMock()
        mock_client_factory.return_value = mock_client
        mock_client.search.return_value = []

        query_vectors = {"semantic_vector": [0.1] * 3072}

        execute_vector_search(query_vectors, odata_filter="scene_type eq 'portrait'")

        call_kwargs = mock_client.search.call_args
        assert call_kwargs.kwargs["filter"] == "scene_type eq 'portrait'"


class TestExecuteHybridSearch:
    """Test the legacy hybrid search function."""

    @patch("ai_search.retrieval.search.get_search_client")
    @patch("ai_search.retrieval.search.load_config")
    def test_builds_vector_queries(
        self,
        mock_config: MagicMock,
        mock_client_factory: MagicMock,
    ) -> None:
        """Should build VectorizedQuery objects with correct weights."""
        from ai_search.config import AppConfig

        mock_config.return_value = AppConfig()

        mock_client = MagicMock()
        mock_client_factory.return_value = mock_client

        # Mock search results
        mock_result = {
            "image_id": "test-1",
            "generation_prompt": "test",
            "@search.score": 0.95,
        }
        mock_client.search.return_value = [mock_result]

        query_vectors = {
            "semantic_vector": [0.1] * 3072,
            "structural_vector": [0.2] * 1024,
            "style_vector": [0.3] * 512,
        }

        results, relevance = execute_hybrid_search("test query", query_vectors)

        assert len(results) == 1
        assert results[0]["search_score"] == 1.0  # normalized: single result → 1.0
        assert relevance is None
        mock_client.search.assert_called_once()

        call_kwargs = mock_client.search.call_args
        assert call_kwargs.kwargs["search_text"] == "test query"
        assert len(call_kwargs.kwargs["vector_queries"]) == 3

    @patch("ai_search.retrieval.search.get_search_client")
    @patch("ai_search.retrieval.search.load_config")
    def test_includes_image_vector_query(
        self,
        mock_config: MagicMock,
        mock_client_factory: MagicMock,
    ) -> None:
        """Should include image_vector query when present."""
        from ai_search.config import AppConfig

        mock_config.return_value = AppConfig()

        mock_client = MagicMock()
        mock_client_factory.return_value = mock_client
        mock_client.search.return_value = []

        query_vectors = {
            "semantic_vector": [0.1] * 3072,
            "structural_vector": [0.2] * 1024,
            "style_vector": [0.3] * 512,
            "image_vector": [0.4] * 1024,
        }

        execute_hybrid_search("test query", query_vectors)

        call_kwargs = mock_client.search.call_args
        assert len(call_kwargs.kwargs["vector_queries"]) == 4

        # Verify image vector query has correct weight
        image_query = call_kwargs.kwargs["vector_queries"][3]
        assert image_query.fields == "image_vector"
