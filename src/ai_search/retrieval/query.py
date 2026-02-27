"""Query embedding generation for text and image queries."""

from __future__ import annotations

import asyncio

from ai_search.clients import get_openai_client
from ai_search.config import load_config
from ai_search.embeddings.encoder import embed_text
from ai_search.embeddings.image import embed_image, embed_text_for_image_search

STRUCTURAL_PROMPT = (
    "Describe the spatial composition, layout, and geometric structure "
    "implied by this query. Focus on object positioning, "
    "foreground/background, and lines of composition. "
    "Write 1-2 sentences."
)

STYLE_PROMPT = (
    "Describe the artistic style, color palette, lighting, and visual "
    "treatment implied by this query. Write 1-2 sentences."
)


async def generate_query_vectors(
    query_text: str,
    query_image_url: str | None = None,
) -> dict[str, list[float]]:
    """Generate semantic, structural, and style query vectors from text.

    Uses the raw query for semantic embedding, and LLM-generated descriptions
    for structural and style embeddings.
    """
    config = load_config()
    client = get_openai_client()

    # Generate structural and style descriptions via LLM
    structural_response = client.chat.completions.create(
        model=config.models.llm_model,
        messages=[
            {"role": "system", "content": STRUCTURAL_PROMPT},
            {"role": "user", "content": query_text},
        ],
        temperature=0.2,
        max_tokens=200,
    )
    structural_desc = structural_response.choices[0].message.content or query_text

    style_response = client.chat.completions.create(
        model=config.models.llm_model,
        messages=[
            {"role": "system", "content": STYLE_PROMPT},
            {"role": "user", "content": query_text},
        ],
        temperature=0.2,
        max_tokens=200,
    )
    style_desc = style_response.choices[0].message.content or query_text

    dims = config.index.vector_dimensions

    # Embed all three in parallel
    semantic_vec, structural_vec, style_vec = await asyncio.gather(
        embed_text(query_text, dimensions=dims.semantic),
        embed_text(structural_desc, dimensions=dims.structural),
        embed_text(style_desc, dimensions=dims.style),
    )

    # Image-space vector (for cross-modal matching)
    if query_image_url:
        image_vec = await embed_image(image_url=query_image_url)
    else:
        image_vec = await embed_text_for_image_search(query_text)

    return {
        "semantic_vector": semantic_vec,
        "structural_vector": structural_vec,
        "style_vector": style_vec,
        "image_vector": image_vec,
    }


def generate_query_vectors_sync(
    query_text: str,
    query_image_url: str | None = None,
) -> dict[str, list[float]]:
    """Synchronous wrapper for query vector generation."""
    return asyncio.run(generate_query_vectors(query_text, query_image_url))
