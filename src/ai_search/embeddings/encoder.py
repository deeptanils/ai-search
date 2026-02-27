"""Base embedding encoder using text-embedding-3-large with Matryoshka dimensions."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ai_search.clients import get_async_openai_client
from ai_search.config import load_config

if TYPE_CHECKING:
    from openai import AsyncAzureOpenAI


async def embed_texts(
    texts: list[str],
    dimensions: int,
    client: AsyncAzureOpenAI | None = None,
) -> list[list[float]]:
    """Embed a batch of texts at the specified dimensionality."""
    if not texts:
        return []

    config = load_config()
    _client = client or get_async_openai_client()
    chunk_size = config.batch.embedding_chunk_size
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), chunk_size):
        chunk = texts[i : i + chunk_size]
        response = await _client.embeddings.create(
            model=config.models.embedding_model,
            input=chunk,
            dimensions=dimensions,
        )
        all_embeddings.extend([item.embedding for item in response.data])

    return all_embeddings


async def embed_text(
    text: str,
    dimensions: int,
    client: AsyncAzureOpenAI | None = None,
) -> list[float]:
    """Embed a single text at the specified dimensionality."""
    results = await embed_texts([text], dimensions, client)
    return results[0]


def embed_text_sync(text: str, dimensions: int) -> list[float]:
    """Synchronous wrapper for single text embedding."""
    return asyncio.run(embed_text(text, dimensions))
