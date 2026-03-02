"""Tests for query vector generation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_search.retrieval.query import generate_image_query_vectors, generate_query_vectors


class TestGenerateQueryVectors:
    """Test the generate_query_vectors function."""

    @pytest.mark.asyncio()
    @patch("ai_search.retrieval.query.embed_text", new_callable=AsyncMock)
    @patch("ai_search.retrieval.query.get_openai_client")
    @patch("ai_search.retrieval.query.load_config")
    async def test_returns_three_text_vector_keys(
        self,
        mock_config: MagicMock,
        mock_client: MagicMock,
        mock_embed_text: AsyncMock,
    ) -> None:
        """Should return dict with semantic, structural, and style vectors only."""
        from ai_search.config import AppConfig

        mock_config.return_value = AppConfig()

        # LLM responses (sync)
        llm_response = MagicMock()
        llm_response.choices = [MagicMock()]
        llm_response.choices[0].message.content = "test description"
        mock_client.return_value.chat.completions.create.return_value = llm_response

        # embed_text returns dimension-aware vectors
        mock_embed_text.side_effect = lambda text, dimensions: [0.1] * dimensions

        result = await generate_query_vectors("sunset photo")

        assert set(result.keys()) == {
            "semantic_vector",
            "structural_vector",
            "style_vector",
        }
        assert len(result["semantic_vector"]) == 3072
        assert len(result["structural_vector"]) == 1024
        assert len(result["style_vector"]) == 512

    @pytest.mark.asyncio()
    @patch("ai_search.retrieval.query.embed_text", new_callable=AsyncMock)
    @patch("ai_search.retrieval.query.get_openai_client")
    @patch("ai_search.retrieval.query.load_config")
    async def test_uses_llm_for_structural_and_style(
        self,
        mock_config: MagicMock,
        mock_client: MagicMock,
        mock_embed_text: AsyncMock,
    ) -> None:
        """Should call LLM twice for structural and style descriptions."""
        from ai_search.config import AppConfig

        mock_config.return_value = AppConfig()

        llm_response = MagicMock()
        llm_response.choices = [MagicMock()]
        llm_response.choices[0].message.content = "test description"
        mock_client.return_value.chat.completions.create.return_value = llm_response

        mock_embed_text.side_effect = lambda text, dimensions: [0.1] * dimensions

        await generate_query_vectors("sunset photo")

        # Two LLM calls: one for structural, one for style
        assert mock_client.return_value.chat.completions.create.call_count == 2

    @pytest.mark.asyncio()
    @patch("ai_search.retrieval.query.embed_text", new_callable=AsyncMock)
    @patch("ai_search.retrieval.query.get_openai_client")
    @patch("ai_search.retrieval.query.load_config")
    async def test_no_image_vector_in_text_mode(
        self,
        mock_config: MagicMock,
        mock_client: MagicMock,
        mock_embed_text: AsyncMock,
    ) -> None:
        """Should not include image_vector in text query vectors."""
        from ai_search.config import AppConfig

        mock_config.return_value = AppConfig()

        llm_response = MagicMock()
        llm_response.choices = [MagicMock()]
        llm_response.choices[0].message.content = "test description"
        mock_client.return_value.chat.completions.create.return_value = llm_response

        mock_embed_text.side_effect = lambda text, dimensions: [0.1] * dimensions

        result = await generate_query_vectors("sunset photo")

        assert "image_vector" not in result


class TestGenerateImageQueryVectors:
    """Test the generate_image_query_vectors function."""

    @pytest.mark.asyncio()
    @patch("ai_search.retrieval.query.embed_image", new_callable=AsyncMock)
    @patch("ai_search.retrieval.query.embed_text", new_callable=AsyncMock)
    @patch("ai_search.retrieval.query.get_openai_client")
    @patch("ai_search.retrieval.query.load_config")
    async def test_returns_all_four_vector_keys(
        self,
        mock_config: MagicMock,
        mock_client: MagicMock,
        mock_embed_text: AsyncMock,
        mock_embed_image: AsyncMock,
    ) -> None:
        """Should return dict with all four vector keys."""
        from ai_search.config import AppConfig
        from ai_search.retrieval.query import QueryImageDescriptions

        mock_config.return_value = AppConfig()

        # GPT-4o structured output
        parsed = QueryImageDescriptions(
            semantic_description="A sunset over the ocean with golden light.",
            structural_description="Horizon line at center with ocean below.",
            style_description="Warm golden tones with soft diffused lighting.",
        )
        parse_response = MagicMock()
        parse_response.choices = [MagicMock()]
        parse_response.choices[0].message.parsed = parsed
        mock_client.return_value.beta.chat.completions.parse.return_value = parse_response

        mock_embed_text.side_effect = lambda text, dimensions: [0.1] * dimensions
        mock_embed_image.return_value = [0.3] * 1024

        result = await generate_image_query_vectors(b"fake-image-bytes")

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
    @patch("ai_search.retrieval.query.embed_image", new_callable=AsyncMock)
    @patch("ai_search.retrieval.query.embed_text", new_callable=AsyncMock)
    @patch("ai_search.retrieval.query.get_openai_client")
    @patch("ai_search.retrieval.query.load_config")
    async def test_calls_gpt4o_for_descriptions(
        self,
        mock_config: MagicMock,
        mock_client: MagicMock,
        mock_embed_text: AsyncMock,
        mock_embed_image: AsyncMock,
    ) -> None:
        """Should call GPT-4o structured output to extract descriptions."""
        from ai_search.config import AppConfig
        from ai_search.retrieval.query import QueryImageDescriptions

        mock_config.return_value = AppConfig()

        parsed = QueryImageDescriptions(
            semantic_description="Test semantic",
            structural_description="Test structural",
            style_description="Test style",
        )
        parse_response = MagicMock()
        parse_response.choices = [MagicMock()]
        parse_response.choices[0].message.parsed = parsed
        mock_client.return_value.beta.chat.completions.parse.return_value = parse_response

        mock_embed_text.side_effect = lambda text, dimensions: [0.1] * dimensions
        mock_embed_image.return_value = [0.3] * 1024

        await generate_image_query_vectors(b"fake-image-bytes")

        mock_client.return_value.beta.chat.completions.parse.assert_called_once()

    @pytest.mark.asyncio()
    @patch("ai_search.retrieval.query.embed_image", new_callable=AsyncMock)
    @patch("ai_search.retrieval.query.embed_text", new_callable=AsyncMock)
    @patch("ai_search.retrieval.query.get_openai_client")
    @patch("ai_search.retrieval.query.load_config")
    async def test_raises_on_no_parsed_output(
        self,
        mock_config: MagicMock,
        mock_client: MagicMock,
        mock_embed_text: AsyncMock,
        mock_embed_image: AsyncMock,
    ) -> None:
        """Should raise ValueError when GPT-4o returns no parsed descriptions."""
        from ai_search.config import AppConfig

        mock_config.return_value = AppConfig()

        parse_response = MagicMock()
        parse_response.choices = [MagicMock()]
        parse_response.choices[0].message.parsed = None
        mock_client.return_value.beta.chat.completions.parse.return_value = parse_response

        with pytest.raises(ValueError, match="no parsed descriptions"):
            await generate_image_query_vectors(b"fake-image-bytes")
