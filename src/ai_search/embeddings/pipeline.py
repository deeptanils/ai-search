"""Embedding pipeline orchestrator — generates all vectors for an image."""

from __future__ import annotations

import asyncio
from typing import Any

from ai_search.embeddings.image import embed_image
from ai_search.embeddings.semantic import generate_semantic_vector
from ai_search.embeddings.structural import generate_structural_vector
from ai_search.embeddings.style import generate_style_vector
from ai_search.models import ImageExtraction, ImageVectors


async def generate_all_vectors(
    extraction: ImageExtraction,
    image_url: str | None = None,
    image_bytes: bytes | None = None,
) -> ImageVectors:
    """Generate all embedding vectors for an extracted image.

    Text embeddings (semantic, structural, style) run in parallel since
    they use text-embedding-3-large.  The image embedding (embed-v-4-0)
    runs separately afterwards to avoid exhausting the S0 rate limit.
    """
    # Step 1: text embeddings in parallel (text-embedding-3-large — separate model)
    semantic_task = generate_semantic_vector(extraction.semantic_description)
    structural_task = generate_structural_vector(extraction.structural_description)
    style_task = generate_style_vector(extraction.style_description)

    text_results = await asyncio.gather(semantic_task, structural_task, style_task)

    # Step 2: image embedding separately (embed-v-4-0 — rate-limited on S0)
    has_image = bool(image_url or image_bytes)
    image_vector: list[float] = []
    if has_image:
        image_vector = await embed_image(image_url=image_url, image_bytes=image_bytes)

    return ImageVectors(
        semantic_vector=text_results[0],
        structural_vector=text_results[1],
        style_vector=text_results[2],
        image_vector=image_vector,
    )
