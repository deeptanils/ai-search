"""Style embedding generation (512 dimensions)."""

from __future__ import annotations

from ai_search.config import load_config
from ai_search.embeddings.encoder import embed_text


async def generate_style_vector(description: str) -> list[float]:
    """Generate a style embedding from a style description."""
    config = load_config()
    return await embed_text(description, dimensions=config.index.vector_dimensions.style)
