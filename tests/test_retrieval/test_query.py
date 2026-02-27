"""Tests for query vector generation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_search.retrieval.query import generate_query_vectors


class TestGenerateQueryVectors:
    """Test the generate_query_vectors function."""

    @pytest.mark.asyncio()
    @patch("ai_search.retrieval.query.embed_text_for_image_search", new_callable=AsyncMock)
    @patch("ai_search.retrieval.query.embed_text", new_callable=AsyncMock)
    @patch("ai_search.retrieval.query.get_openai_client")
    @patch("ai_search.retrieval.query.load_config")
    async def test_returns_all_vector_keys(
        self,
        mock_config: MagicMock,
        mock_client: MagicMock,
        mock_embed_text: AsyncMock,
        mock_embed_text_image: AsyncMock,
    ) -> None:
        """Should return dict with all four vector keys including image_vector."""
        from ai_search.config import AppConfig

        mock_config.return_value = AppConfig()

        # LLM responses (sync)
        llm_response = MagicMock()
        llm_response.choices = [MagicMock()]
        llm_response.choices[0].message.content = "test description"
        mock_client.return_value.chat.completions.create.return_value = llm_response

        # embed_text returns dimension-aware vectors
        mock_embed_text.side_effect = lambda text, dimensions: [0.1] * dimensions

        # embed_text_for_image_search returns 1024-dim
        mock_embed_text_image.return_value = [0.2] * 1024

        result = await generate_query_vectors("sunset photo")

        assert set(result.keys()) == {
            "semantic_vector",
            "structural_vector",
            "style_vector",
            "image_vector",
        }
        assert len(result["semantic_vector"]) == 3072
        assert len(result["structural_vector"]) == 1024
        assert len(result["style_vector"]) == 512
        assert len(result["image_vector"]) == 1024

    @pytest.mark.asyncio()
    @patch("ai_search.retrieval.query.embed_text_for_image_search", new_callable=AsyncMock)
    @patch("ai_search.retrieval.query.embed_image", new_callable=AsyncMock)
    @patch("ai_search.retrieval.query.embed_text", new_callable=AsyncMock)
    @patch("ai_search.retrieval.query.get_openai_client")
    @patch("ai_search.retrieval.query.load_config")
    async def test_image_url_calls_embed_image(
        self,
        mock_config: MagicMock,
        mock_client: MagicMock,
        mock_embed_text: AsyncMock,
        mock_embed_image: AsyncMock,
        mock_embed_text_image: AsyncMock,
    ) -> None:
        """Should call embed_image when query_image_url is provided."""
        from ai_search.config import AppConfig

        mock_config.return_value = AppConfig()

        llm_response = MagicMock()
        llm_response.choices = [MagicMock()]
        llm_response.choices[0].message.content = "test description"
        mock_client.return_value.chat.completions.create.return_value = llm_response

        mock_embed_text.side_effect = lambda text, dimensions: [0.1] * dimensions
        mock_embed_image.return_value = [0.3] * 1024

        result = await generate_query_vectors(
            "sunset photo",
            query_image_url="https://example.com/ref.jpg",
        )

        mock_embed_image.assert_called_once_with(image_url="https://example.com/ref.jpg")
        mock_embed_text_image.assert_not_called()
        assert len(result["image_vector"]) == 1024

    @pytest.mark.asyncio()
    @patch("ai_search.retrieval.query.embed_image", new_callable=AsyncMock)
    @patch("ai_search.retrieval.query.embed_text_for_image_search", new_callable=AsyncMock)
    @patch("ai_search.retrieval.query.embed_text", new_callable=AsyncMock)
    @patch("ai_search.retrieval.query.get_openai_client")
    @patch("ai_search.retrieval.query.load_config")
    async def test_text_only_calls_embed_text_for_image_search(
        self,
        mock_config: MagicMock,
        mock_client: MagicMock,
        mock_embed_text: AsyncMock,
        mock_embed_text_image: AsyncMock,
        mock_embed_image: AsyncMock,
    ) -> None:
        """Should call embed_text_for_image_search when no image URL provided."""
        from ai_search.config import AppConfig

        mock_config.return_value = AppConfig()

        llm_response = MagicMock()
        llm_response.choices = [MagicMock()]
        llm_response.choices[0].message.content = "test description"
        mock_client.return_value.chat.completions.create.return_value = llm_response

        mock_embed_text.side_effect = lambda text, dimensions: [0.1] * dimensions
        mock_embed_text_image.return_value = [0.4] * 1024

        result = await generate_query_vectors("sunset photo")

        mock_embed_text_image.assert_called_once_with("sunset photo")
        mock_embed_image.assert_not_called()
        assert len(result["image_vector"]) == 1024
