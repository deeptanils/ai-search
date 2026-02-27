"""Structural embedding generation (1024 dimensions)."""

from __future__ import annotations

from ai_search.config import load_config
from ai_search.embeddings.encoder import embed_text


async def generate_structural_vector(description: str) -> list[float]:
    """Generate a structural embedding from a composition description."""
    config = load_config()
    return await embed_text(description, dimensions=config.index.vector_dimensions.structural)
