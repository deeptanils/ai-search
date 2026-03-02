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

    All four embeddings (semantic, structural, style, image) run in
    parallel via ``asyncio.gather`` for maximum throughput.
    """
    # Build tasks for text embeddings
    tasks: list[Any] = [
        generate_semantic_vector(extraction.semantic_description),
        generate_structural_vector(extraction.structural_description),
        generate_style_vector(extraction.style_description),
    ]

    # Add image embedding task if image data is available
    has_image = bool(image_url or image_bytes)
    if has_image:
        tasks.append(embed_image(image_url=image_url, image_bytes=image_bytes))

    results = await asyncio.gather(*tasks)

    return ImageVectors(
        semantic_vector=results[0],
        structural_vector=results[1],
        style_vector=results[2],
        image_vector=results[3] if has_image else [],
    )
