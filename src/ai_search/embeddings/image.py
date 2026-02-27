"""Image embedding via configurable backend.

Supports:
- Azure Computer Vision 4.0 (Florence) — requires AZURE_CV_* env vars
- Azure AI Foundry models (e.g., Cohere Embed v4) — uses Foundry endpoint

Backend selection is controlled by ``models.image_embedding_model`` in
config.yaml.  Set to ``"azure-cv-florence"`` for Florence or a Foundry
deployment name (e.g., ``"embed-v-4-0"``) for model-inference embeddings.
"""

from __future__ import annotations

import base64
import io

import httpx
import structlog
from PIL import Image

from azure.ai.inference.models import ImageEmbeddingInput

from ai_search.clients import get_cv_client, get_foundry_embed_client, get_foundry_image_embed_client
from ai_search.config import load_config, load_cv_secrets

logger = structlog.get_logger(__name__)

_FLORENCE_BACKEND = "azure-cv-florence"


# ---------------------------------------------------------------------------
# Shared validation
# ---------------------------------------------------------------------------


def _validate_vector(
    vector: list[float] | None,
    expected_dims: int,
    endpoint_name: str,
    response_keys: list[str],
) -> list[float]:
    """Validate embedding vector dimensions.

    Args:
        vector: Embedding vector from the API response.
        expected_dims: Required number of dimensions.
        endpoint_name: Identifier for log/error messages.
        response_keys: Response-level keys for diagnostics.

    Returns:
        The validated vector.

    Raises:
        ValueError: If vector is missing, empty, or wrong dimensions.
    """
    if not vector or len(vector) != expected_dims:
        logger.warning(
            f"{endpoint_name} response invalid",
            expected_dimensions=expected_dims,
            actual_dimensions=len(vector) if vector else None,
            response_keys=response_keys,
        )
        msg = (
            f"{endpoint_name} response invalid "
            f"(expected {expected_dims}-dim vector, "
            f"got {len(vector) if vector else 'None'}): "
            f"keys={response_keys}"
        )
        raise ValueError(msg)

    result: list[float] = vector
    return result


# ---------------------------------------------------------------------------
# Florence backend (Azure Computer Vision 4.0)
# ---------------------------------------------------------------------------


async def _embed_image_florence(
    image_url: str | None = None,
    image_bytes: bytes | None = None,
) -> list[float]:
    """Embed image via Azure Computer Vision 4.0 (Florence)."""
    client = get_cv_client()
    secrets = load_cv_secrets()
    params = {"api-version": secrets.api_version, "model-version": secrets.model_version}

    if image_url:
        response = await client.post(
            "/computervision/retrieval:vectorizeImage",
            params=params,
            json={"url": image_url},
        )
    else:
        response = await client.post(
            "/computervision/retrieval:vectorizeImage",
            params=params,
            content=image_bytes,
            headers={"Content-Type": "application/octet-stream"},
        )

    response.raise_for_status()
    data = response.json()
    return _validate_vector(
        data.get("vector"), 1024, "Florence vectorizeImage", list(data.keys()),
    )


async def _embed_text_florence(text: str) -> list[float]:
    """Embed text via Azure Computer Vision 4.0 (Florence)."""
    client = get_cv_client()
    secrets = load_cv_secrets()
    params = {"api-version": secrets.api_version, "model-version": secrets.model_version}

    response = await client.post(
        "/computervision/retrieval:vectorizeText",
        params=params,
        json={"text": text},
    )
    response.raise_for_status()
    data = response.json()
    return _validate_vector(
        data.get("vector"), 1024, "Florence vectorizeText", list(data.keys()),
    )


# ---------------------------------------------------------------------------
# Foundry backend (e.g., Cohere Embed v4)
# ---------------------------------------------------------------------------


# Maximum image dimension before base64 encoding.  Keeping images small
# dramatically reduces the token count sent to embed-v-4-0, which is
# critical for staying within S0-tier rate limits.
_MAX_IMAGE_SIZE = 512


def _resize_image_bytes(raw: bytes, max_size: int = _MAX_IMAGE_SIZE) -> bytes:
    """Resize image to fit within max_size×max_size and re-encode as JPEG."""
    img = Image.open(io.BytesIO(raw))
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


async def _image_to_data_uri(
    image_url: str | None = None,
    image_bytes: bytes | None = None,
) -> str:
    """Convert an image URL or bytes to a base64 data URI string.

    Images are resized to at most 512×512 to reduce token consumption
    and avoid S0-tier rate limits on embed-v-4-0.
    """
    if image_bytes is not None:
        small = _resize_image_bytes(image_bytes)
        b64 = base64.b64encode(small).decode()
        return f"data:image/jpeg;base64,{b64}"

    assert image_url is not None  # guaranteed by caller
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as dl:
        resp = await dl.get(image_url)
        resp.raise_for_status()
    small = _resize_image_bytes(resp.content)
    b64 = base64.b64encode(small).decode()
    return f"data:image/jpeg;base64,{b64}"


async def _embed_image_foundry(
    model: str,
    dimensions: int,
    image_url: str | None = None,
    image_bytes: bytes | None = None,
) -> list[float]:
    """Embed image via Azure AI Inference ImageEmbeddingsClient.

    Uses the ``/images/embeddings`` route with ``ImageEmbeddingInput`` to
    ensure the model processes the image visually rather than tokenizing
    the base64 string as text.
    """
    data_uri = await _image_to_data_uri(image_url=image_url, image_bytes=image_bytes)
    client = get_foundry_image_embed_client()
    response = await client.embed(
        input=[ImageEmbeddingInput(image=data_uri)],
        model=model,
        dimensions=dimensions,
    )
    vector = list(response.data[0].embedding)
    return _validate_vector(
        vector, dimensions, f"Foundry {model} image embedding", [],
    )


async def _embed_text_foundry(model: str, text: str, dimensions: int) -> list[float]:
    """Embed text via Azure AI Inference EmbeddingsClient."""
    client = get_foundry_embed_client()
    response = await client.embed(
        input=[text],
        model=model,
        dimensions=dimensions,
    )
    vector = list(response.data[0].embedding)
    return _validate_vector(
        vector, dimensions, f"Foundry {model} text embedding", [],
    )


# ---------------------------------------------------------------------------
# Public API — routes to configured backend
# ---------------------------------------------------------------------------


async def embed_image(
    image_url: str | None = None,
    image_bytes: bytes | None = None,
) -> list[float]:
    """Embed an image into a vector via the configured backend.

    The backend is selected by ``models.image_embedding_model`` in
    config.yaml.  Use ``"azure-cv-florence"`` for Florence or any
    Foundry deployment name for model-inference embeddings.

    Args:
        image_url: Public URL of the image.
        image_bytes: Raw image bytes (JPEG/PNG).

    Returns:
        Float vector in the model's shared image-text space.

    Raises:
        ValueError: If neither input is provided or the response is invalid.
        httpx.HTTPStatusError: If the embedding API returns an error.
    """
    if not image_url and not image_bytes:
        msg = "Either image_url or image_bytes must be provided"
        raise ValueError(msg)

    config = load_config()
    backend = config.models.image_embedding_model

    if backend == _FLORENCE_BACKEND:
        result = await _embed_image_florence(image_url=image_url, image_bytes=image_bytes)
    else:
        dims = config.index.vector_dimensions.image
        result = await _embed_image_foundry(
            model=backend,
            dimensions=dims,
            image_url=image_url,
            image_bytes=image_bytes,
        )

    logger.info("Image embedded", backend=backend, dimensions=len(result))
    return result


async def embed_text_for_image_search(text: str) -> list[float]:
    """Embed text into the image-embedding space for cross-modal search.

    Uses the configured image embedding model so the resulting vector
    is comparable (cosine similarity) with image vectors from
    ``embed_image()``.

    Args:
        text: Query text to embed in the image space.

    Returns:
        Float vector in the model's shared image-text space.

    Raises:
        ValueError: If the API response is invalid.
        httpx.HTTPStatusError: If the embedding API returns an error.
    """
    config = load_config()
    backend = config.models.image_embedding_model

    if backend == _FLORENCE_BACKEND:
        result = await _embed_text_florence(text)
    else:
        dims = config.index.vector_dimensions.image
        result = await _embed_text_foundry(model=backend, text=text, dimensions=dims)

    logger.info("Text embedded for image search", backend=backend, dimensions=len(result))
    return result
