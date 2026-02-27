"""Tests for the embedding pipeline orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ai_search.models import ImageExtraction, ImageVectors


class TestGenerateAllVectors:
    """Test the generate_all_vectors function."""

    @pytest.mark.asyncio()
    @patch("ai_search.embeddings.pipeline.generate_style_vector", new_callable=AsyncMock)
    @patch("ai_search.embeddings.pipeline.generate_structural_vector", new_callable=AsyncMock)
    @patch("ai_search.embeddings.pipeline.generate_semantic_vector", new_callable=AsyncMock)
    async def test_parallel_execution(
        self,
        mock_semantic: AsyncMock,
        mock_structural: AsyncMock,
        mock_style: AsyncMock,
        sample_extraction: ImageExtraction,
    ) -> None:
        """All three embedding tasks should be invoked without image input."""
        mock_semantic.return_value = [0.1] * 3072
        mock_structural.return_value = [0.2] * 1024
        mock_style.return_value = [0.3] * 512

        from ai_search.embeddings.pipeline import generate_all_vectors

        result = await generate_all_vectors(sample_extraction)

        assert isinstance(result, ImageVectors)
        assert len(result.semantic_vector) == 3072
        assert len(result.structural_vector) == 1024
        assert len(result.style_vector) == 512
        assert result.image_vector == []

        mock_semantic.assert_called_once_with(sample_extraction.semantic_description)
        mock_structural.assert_called_once_with(sample_extraction.structural_description)
        mock_style.assert_called_once_with(sample_extraction.style_description)

    @pytest.mark.asyncio()
    @patch("ai_search.embeddings.pipeline.embed_image", new_callable=AsyncMock)
    @patch("ai_search.embeddings.pipeline.generate_style_vector", new_callable=AsyncMock)
    @patch("ai_search.embeddings.pipeline.generate_structural_vector", new_callable=AsyncMock)
    @patch("ai_search.embeddings.pipeline.generate_semantic_vector", new_callable=AsyncMock)
    async def test_with_image_url(
        self,
        mock_semantic: AsyncMock,
        mock_structural: AsyncMock,
        mock_style: AsyncMock,
        mock_embed_image: AsyncMock,
        sample_extraction: ImageExtraction,
    ) -> None:
        """Image vector should be populated when image_url is provided."""
        mock_semantic.return_value = [0.1] * 3072
        mock_structural.return_value = [0.2] * 1024
        mock_style.return_value = [0.3] * 512
        mock_embed_image.return_value = [0.4] * 1024

        from ai_search.embeddings.pipeline import generate_all_vectors

        result = await generate_all_vectors(
            sample_extraction,
            image_url="https://example.com/image.jpg",
        )

        assert isinstance(result, ImageVectors)
        assert len(result.image_vector) == 1024
        mock_embed_image.assert_called_once_with(
            image_url="https://example.com/image.jpg",
            image_bytes=None,
        )
