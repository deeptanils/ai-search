"""Tests for the base embedding encoder."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_search.embeddings.encoder import embed_text, embed_texts


class TestEmbedTexts:
    """Test the embed_texts function."""

    @pytest.mark.asyncio()
    async def test_empty_input(self) -> None:
        """Empty input should return empty list."""
        result = await embed_texts([], dimensions=3072)
        assert result == []

    @pytest.mark.asyncio()
    @patch("ai_search.embeddings.encoder.load_config")
    async def test_single_text(self, mock_config: MagicMock) -> None:
        """Single text should produce one embedding."""
        config = MagicMock()
        config.models.embedding_model = "text-embedding-3-large"
        config.batch.embedding_chunk_size = 2048
        mock_config.return_value = config

        mock_client = AsyncMock()
        embedding_item = MagicMock()
        embedding_item.embedding = [0.1] * 256
        mock_response = MagicMock()
        mock_response.data = [embedding_item]
        mock_client.embeddings.create.return_value = mock_response

        result = await embed_texts(["test text"], dimensions=256, client=mock_client)

        assert len(result) == 1
        assert len(result[0]) == 256
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-large",
            input=["test text"],
            dimensions=256,
        )

    @pytest.mark.asyncio()
    @patch("ai_search.embeddings.encoder.load_config")
    async def test_batching(self, mock_config: MagicMock) -> None:
        """Texts exceeding chunk_size should be batched."""
        config = MagicMock()
        config.models.embedding_model = "text-embedding-3-large"
        config.batch.embedding_chunk_size = 2  # Force batching
        mock_config.return_value = config

        mock_client = AsyncMock()

        def make_response(texts: list[str]) -> MagicMock:
            items = []
            for _ in texts:
                item = MagicMock()
                item.embedding = [0.1] * 256
                items.append(item)
            resp = MagicMock()
            resp.data = items
            return resp

        # Return different response for each batch
        mock_client.embeddings.create.side_effect = [
            make_response(["a", "b"]),
            make_response(["c"]),
        ]

        result = await embed_texts(["a", "b", "c"], dimensions=256, client=mock_client)
        assert len(result) == 3
        assert mock_client.embeddings.create.call_count == 2


class TestEmbedText:
    """Test the embed_text function."""

    @pytest.mark.asyncio()
    @patch("ai_search.embeddings.encoder.load_config")
    async def test_single_embed(self, mock_config: MagicMock) -> None:
        """Single text embedding should return a flat list."""
        config = MagicMock()
        config.models.embedding_model = "test-model"
        config.batch.embedding_chunk_size = 2048
        mock_config.return_value = config

        mock_client = AsyncMock()
        embedding_item = MagicMock()
        embedding_item.embedding = [0.5] * 512
        mock_response = MagicMock()
        mock_response.data = [embedding_item]
        mock_client.embeddings.create.return_value = mock_response

        result = await embed_text("hello", dimensions=512, client=mock_client)
        assert len(result) == 512
