"""Tests for the image embedding module (Florence and Foundry backends)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from ai_search.embeddings.image import embed_image, embed_text_for_image_search

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _default_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default to Florence backend for all image embedding tests."""
    config = MagicMock()
    config.models.image_embedding_model = "azure-cv-florence"
    config.index.vector_dimensions.image = 1024
    monkeypatch.setattr("ai_search.embeddings.image.load_config", lambda: config)


@pytest.fixture()
def mock_cv_client(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Mock the Florence CV client and secrets."""
    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.json.return_value = {"vector": [0.1] * 1024}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    secrets = MagicMock()
    secrets.api_version = "2024-02-01"

    monkeypatch.setattr("ai_search.embeddings.image.get_cv_client", lambda: mock_client)
    monkeypatch.setattr("ai_search.embeddings.image.load_cv_secrets", lambda: secrets)
    return mock_client


@pytest.fixture()
def mock_foundry_backend(monkeypatch: pytest.MonkeyPatch) -> dict[str, AsyncMock]:
    """Mock Foundry backend: config, image download, and embedding clients."""
    config = MagicMock()
    config.models.image_embedding_model = "embed-v-4-0"
    config.index.vector_dimensions.image = 1024
    monkeypatch.setattr("ai_search.embeddings.image.load_config", lambda: config)

    # Mock _image_to_data_uri so we don't download anything
    monkeypatch.setattr(
        "ai_search.embeddings.image._image_to_data_uri",
        AsyncMock(return_value="data:image/jpeg;base64,AAAA"),
    )

    # Mock Azure AI Inference ImageEmbeddingsClient (image route)
    embedding_obj = MagicMock()
    embedding_obj.embedding = [0.2] * 1024
    embed_response = MagicMock()
    embed_response.data = [embedding_obj]
    image_embed_client = AsyncMock()
    image_embed_client.embed = AsyncMock(return_value=embed_response)
    monkeypatch.setattr(
        "ai_search.embeddings.image.get_foundry_image_embed_client",
        lambda: image_embed_client,
    )

    # Mock Azure AI Inference EmbeddingsClient (text route, for text queries)
    text_embed_client = AsyncMock()
    text_embed_client.embed = AsyncMock(return_value=embed_response)
    monkeypatch.setattr(
        "ai_search.embeddings.image.get_foundry_embed_client",
        lambda: text_embed_client,
    )

    return {
        "image_embed_client": image_embed_client,
        "text_embed_client": text_embed_client,
    }


# ---------------------------------------------------------------------------
# Florence backend tests
# ---------------------------------------------------------------------------


class TestEmbedImage:
    """Test the embed_image function."""

    @pytest.mark.asyncio()
    async def test_embed_image_url(self, mock_cv_client: AsyncMock) -> None:
        """Should return 1024-dim vector from image URL."""
        result = await embed_image(image_url="https://example.com/test.jpg")

        assert len(result) == 1024
        assert all(isinstance(v, float) for v in result)
        mock_cv_client.post.assert_called_once()
        call_kwargs = mock_cv_client.post.call_args
        assert "vectorizeImage" in call_kwargs.args[0]
        assert call_kwargs.kwargs["json"] == {"url": "https://example.com/test.jpg"}

    @pytest.mark.asyncio()
    async def test_embed_image_bytes(self, mock_cv_client: AsyncMock) -> None:
        """Should return 1024-dim vector from raw image bytes."""
        test_bytes = b"\x89PNG\r\n\x1a\nfake_image_data"
        result = await embed_image(image_bytes=test_bytes)

        assert len(result) == 1024
        mock_cv_client.post.assert_called_once()
        call_kwargs = mock_cv_client.post.call_args
        assert call_kwargs.kwargs["content"] == test_bytes

    @pytest.mark.asyncio()
    async def test_embed_image_no_input_raises(self) -> None:
        """Should raise ValueError when neither URL nor bytes provided."""
        with pytest.raises(ValueError, match="Either image_url or image_bytes must be provided"):
            await embed_image()

    @pytest.mark.asyncio()
    async def test_embed_image_api_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should propagate HTTPStatusError on API failure."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=httpx.Request("POST", "https://test.com"),
            response=httpx.Response(500),
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        secrets = MagicMock()
        secrets.api_version = "2024-02-01"

        monkeypatch.setattr("ai_search.embeddings.image.get_cv_client", lambda: mock_client)
        monkeypatch.setattr("ai_search.embeddings.image.load_cv_secrets", lambda: secrets)

        with pytest.raises(httpx.HTTPStatusError):
            await embed_image(image_url="https://example.com/bad.jpg")


class TestEmbedTextForImageSearch:
    """Test the embed_text_for_image_search function."""

    @pytest.mark.asyncio()
    async def test_embed_text_for_image_search(self, mock_cv_client: AsyncMock) -> None:
        """Should return 1024-dim vector from text via Florence vectorizeText."""
        result = await embed_text_for_image_search("a woman in a red dress")

        assert len(result) == 1024
        assert all(isinstance(v, float) for v in result)
        mock_cv_client.post.assert_called_once()
        call_kwargs = mock_cv_client.post.call_args
        assert "vectorizeText" in call_kwargs.args[0]
        assert call_kwargs.kwargs["json"] == {"text": "a woman in a red dress"}


class TestEmbedImageValidation:
    """Test embed_image response validation."""

    @pytest.mark.asyncio()
    async def test_missing_vector_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should raise ValueError when response lacks 'vector' key."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.json.return_value = {"modelVersion": "2023-04-15"}
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        secrets = MagicMock()
        secrets.api_version = "2024-02-01"
        monkeypatch.setattr("ai_search.embeddings.image.get_cv_client", lambda: mock_client)
        monkeypatch.setattr("ai_search.embeddings.image.load_cv_secrets", lambda: secrets)

        with pytest.raises(ValueError, match="Florence vectorizeImage response invalid"):
            await embed_image(image_url="https://example.com/test.jpg")

    @pytest.mark.asyncio()
    async def test_empty_vector_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should raise ValueError when response vector is empty."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.json.return_value = {"vector": []}
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        secrets = MagicMock()
        secrets.api_version = "2024-02-01"
        monkeypatch.setattr("ai_search.embeddings.image.get_cv_client", lambda: mock_client)
        monkeypatch.setattr("ai_search.embeddings.image.load_cv_secrets", lambda: secrets)

        with pytest.raises(ValueError, match="Florence vectorizeImage response invalid"):
            await embed_image(image_url="https://example.com/test.jpg")

    @pytest.mark.asyncio()
    async def test_wrong_dimensions_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should raise ValueError when vector has wrong dimensions."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.json.return_value = {"vector": [0.1] * 512}
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        secrets = MagicMock()
        secrets.api_version = "2024-02-01"
        monkeypatch.setattr("ai_search.embeddings.image.get_cv_client", lambda: mock_client)
        monkeypatch.setattr("ai_search.embeddings.image.load_cv_secrets", lambda: secrets)

        with pytest.raises(ValueError, match="expected 1024-dim vector, got 512"):
            await embed_image(image_url="https://example.com/test.jpg")


class TestEmbedTextForImageSearchValidation:
    """Test embed_text_for_image_search response validation."""

    @pytest.mark.asyncio()
    async def test_missing_vector_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should raise ValueError when response lacks 'vector' key."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.json.return_value = {"error": "unexpected"}
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        secrets = MagicMock()
        secrets.api_version = "2024-02-01"
        monkeypatch.setattr("ai_search.embeddings.image.get_cv_client", lambda: mock_client)
        monkeypatch.setattr("ai_search.embeddings.image.load_cv_secrets", lambda: secrets)

        with pytest.raises(ValueError, match="Florence vectorizeText response invalid"):
            await embed_text_for_image_search("sunset over mountains")

    @pytest.mark.asyncio()
    async def test_wrong_dimensions_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should raise ValueError when vector has wrong dimensions."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.json.return_value = {"vector": [0.1] * 256}
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        secrets = MagicMock()
        secrets.api_version = "2024-02-01"
        monkeypatch.setattr("ai_search.embeddings.image.get_cv_client", lambda: mock_client)
        monkeypatch.setattr("ai_search.embeddings.image.load_cv_secrets", lambda: secrets)

        with pytest.raises(ValueError, match="expected 1024-dim vector, got 256"):
            await embed_text_for_image_search("sunset over mountains")


class TestSecretsValidation:
    """Test secrets misconfiguration detection."""

    async def test_missing_secrets_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should raise ValueError when CV secrets are not configured."""
        from ai_search.clients import get_cv_client

        secrets = MagicMock()
        secrets.endpoint = None
        secrets.api_key = None
        monkeypatch.setattr("ai_search.clients.load_cv_secrets", lambda: secrets)
        get_cv_client.cache_clear()

        try:
            with pytest.raises(ValueError, match="Azure Computer Vision secrets not configured"):
                get_cv_client()
        finally:
            get_cv_client.cache_clear()


# ---------------------------------------------------------------------------
# Foundry backend tests
# ---------------------------------------------------------------------------


class TestFoundryEmbedImage:
    """Test embed_image with Foundry backend."""

    @pytest.mark.asyncio()
    async def test_embed_image_url(self, mock_foundry_backend: dict[str, AsyncMock]) -> None:
        """Should return 1024-dim vector from image URL via Foundry."""
        result = await embed_image(image_url="https://example.com/test.jpg")

        assert len(result) == 1024
        image_embed_client = mock_foundry_backend["image_embed_client"]
        image_embed_client.embed.assert_called_once()
        call_kwargs = image_embed_client.embed.call_args.kwargs
        assert call_kwargs["model"] == "embed-v-4-0"
        assert call_kwargs["dimensions"] == 1024
        # Input should be ImageEmbeddingInput, not a plain string
        inputs = call_kwargs["input"]
        assert len(inputs) == 1
        assert inputs[0].image == "data:image/jpeg;base64,AAAA"

    @pytest.mark.asyncio()
    async def test_embed_image_bytes(self, mock_foundry_backend: dict[str, AsyncMock]) -> None:
        """Should convert bytes to data URI and embed via Foundry."""
        result = await embed_image(image_bytes=b"fake-png-data")

        assert len(result) == 1024
        image_embed_client = mock_foundry_backend["image_embed_client"]
        call_kwargs = image_embed_client.embed.call_args.kwargs
        inputs = call_kwargs["input"]
        assert len(inputs) == 1
        assert inputs[0].image == "data:image/jpeg;base64,AAAA"


class TestFoundryEmbedText:
    """Test embed_text_for_image_search with Foundry backend."""

    @pytest.mark.asyncio()
    async def test_embed_text(self, mock_foundry_backend: dict[str, AsyncMock]) -> None:
        """Should return 1024-dim vector from text via Azure AI Inference SDK."""
        result = await embed_text_for_image_search("woman in red dress")

        assert len(result) == 1024
        text_embed_client = mock_foundry_backend["text_embed_client"]
        text_embed_client.embed.assert_called_once()
        call_kwargs = text_embed_client.embed.call_args.kwargs
        assert call_kwargs["model"] == "embed-v-4-0"
        assert call_kwargs["dimensions"] == 1024


class TestFoundryValidation:
    """Test Foundry backend response validation."""

    @pytest.mark.asyncio()
    async def test_invalid_image_response_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should raise ValueError when Foundry image response has wrong dims."""
        config = MagicMock()
        config.models.image_embedding_model = "embed-v-4-0"
        config.index.vector_dimensions.image = 1024
        monkeypatch.setattr("ai_search.embeddings.image.load_config", lambda: config)

        # Mock _image_to_data_uri
        monkeypatch.setattr(
            "ai_search.embeddings.image._image_to_data_uri",
            AsyncMock(return_value="data:image/jpeg;base64,AAAA"),
        )

        # Mock EmbeddingsClient returning wrong dimensions
        embedding_obj = MagicMock()
        embedding_obj.embedding = [0.1] * 512
        embed_response = MagicMock()
        embed_response.data = [embedding_obj]
        embed_client = AsyncMock()
        embed_client.embed = AsyncMock(return_value=embed_response)
        monkeypatch.setattr("ai_search.embeddings.image.get_foundry_image_embed_client", lambda: embed_client)

        with pytest.raises(ValueError, match="expected 1024-dim vector, got 512"):
            await embed_image(image_url="https://example.com/test.jpg")
