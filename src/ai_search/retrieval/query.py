"""Query embedding generation for text and image queries."""

from __future__ import annotations

import asyncio
import base64
import hashlib

import structlog
from pydantic import BaseModel, Field

from ai_search.clients import get_async_openai_client, get_openai_client
from ai_search.config import load_config
from ai_search.embeddings.encoder import embed_text
from ai_search.embeddings.image import _resize_image_bytes, embed_image

logger = structlog.get_logger(__name__)

STRUCTURAL_PROMPT = (
    "You are a cinematic composition analyst describing the visual "
    "composition of mythological and cultural artwork. Given a search "
    "query, describe the spatial composition as a movie director would "
    "plan the shot.\n\n"
    "Cover: character count and positioning on screen, scale "
    "relationships (giant vs normal, divine vs mortal, army vs "
    "individual), camera angle (low-angle heroic, aerial establishing, "
    "close-up emotional, wide battle panorama, Dutch tilt for tension), "
    "foreground/midground/background layering, action direction and "
    "dynamic lines of movement (flying upward, charging forward, "
    "falling), depth staging (characters in front of armies, weapons "
    "extended toward viewer), focal point placement, symmetry vs "
    "asymmetry.\n\n"
    "Consider Indian mythology art conventions — dramatic divine poses, "
    "multi-figure battle compositions, ceremonial arrangements, "
    "landscape-scale scenes with tiny armies below giant characters. "
    "Write 4-5 detailed sentences."
)

STYLE_PROMPT = (
    "You are a cinematic art director describing the visual style "
    "of mythological and cultural artwork. Given a search query, "
    "describe the artistic style as if directing the visual treatment "
    "for a movie scene.\n\n"
    "Cover: color palette (golden/fiery reds for battle scenes, cool "
    "blues for divine visions, earthy greens for forests, jewel tones "
    "for palace scenes), lighting type (divine glow, dramatic "
    "backlighting, firelight from burning Lanka, moonlight, sunset "
    "silhouettes, volumetric god-rays), texture (detailed ornamental "
    "jewelry, battle-worn armor, silken robes, weathered stone), "
    "atmosphere (misty, smoky battlefield, radiant divine presence, "
    "stormy ocean, celestial clouds), rendering approach (cinematic "
    "VFX, digital painting, photorealistic), cultural art conventions "
    "(Indian miniature style, Mughal-era detailing, traditional "
    "temple mural aesthetics, modern concept art).\n\n"
    "Write 4-5 detailed sentences."
)

QUERY_IMAGE_SYSTEM_PROMPT = (
    "You are an expert cinematic image analyst — think like a movie "
    "director analyzing a movie still. Specialize in mythology, "
    "religion, folklore, and cultural art. Identify characters by "
    "PROPER NAME (Hanuman, Rama, Sita, Ravana, Krishna, Arjuna, "
    "Lakshman, Sugriva, Vibhishana) and name the specific story "
    "episode. If no mythological reference is identifiable, describe "
    "the scene using specific nouns.\n\n"
    "CRITICAL: Focus on dimensions that DISTINGUISH this scene from "
    "other scenes with the same character(s). Two images of Hanuman "
    "should produce very different descriptions if one shows him "
    "flying with a mountain and the other shows him fighting in "
    "Lanka.\n\n"
    "Extract three descriptions:\n\n"
    "1. semantic_description (150 words): Name every character "
    "visible. Describe their specific ACTION (flying, fighting, "
    "kneeling, blessing, lifting, burning, talking, meditating). "
    "List WEAPONS and PROPS held or nearby (gada/mace, bow, arrows, "
    "mountain, torch, chariot, trident, chakra, lotus). Describe "
    "COSTUMES and JEWELRY (armor, crown, royal robes, forest attire, "
    "dhoti, ornaments, sacred thread, anklets, armlets). Note "
    "character FORM or AVATAR (giant Hanuman, multi-headed Ravana, "
    "blue-skinned Rama). Identify the LOCATION (battlefield, palace, "
    "forest, ocean, sky, Lanka, Ayodhya, Kishkindha, Panchavati, "
    "Ashoka Vatika). Name the story EPISODE (Lanka Dahan, Sanjeevani "
    "quest, Setu Bandhan, Sita Haran, Swayamvar, Agni Pariksha, "
    "Bali Vadh). State the EMOTIONAL TONE (fury, devotion, sorrow, "
    "triumph, serenity, determination, anguish).\n\n"
    "2. structural_description (80 words): Spatial composition — "
    "character count and positioning, scale relationships (giant vs "
    "normal, divine vs mortal), camera angle (low-angle heroic, "
    "aerial, close-up, wide establishing shot), foreground/background "
    "separation, action direction and dynamic lines of movement, "
    "focal point placement.\n\n"
    "3. style_description (80 words): Artistic treatment — color "
    "palette (golden, fiery reds, cool blues, earthy greens, jewel "
    "tones), lighting type (divine glow, dramatic backlight, "
    "firelight, moonlight, sunset), texture (detailed ornamental, "
    "painterly, photorealistic), atmosphere (misty, smoky, radiant, "
    "stormy), rendering approach and cultural art conventions."
)


class QueryImageDescriptions(BaseModel):
    """Descriptions extracted from a query image for multi-vector search."""

    semantic_description: str = Field(
        description="150-word scene description: named characters, specific "
        "actions, weapons/props, costumes/jewelry, character form/avatar, "
        "location, story episode, and emotional tone"
    )
    structural_description: str = Field(
        description="80-word spatial composition: character positioning, "
        "scale relationships, camera angle, action direction, focal point"
    )
    style_description: str = Field(
        description="80-word artistic treatment: color palette, lighting "
        "type, texture, atmosphere, rendering approach"
    )


# ---------------------------------------------------------------------------
# Async LLM helpers
# ---------------------------------------------------------------------------


async def _expand_structural(query_text: str) -> str:
    """Generate a structural description from the query via async LLM."""
    config = load_config()
    client = get_async_openai_client()
    response = await client.chat.completions.create(
        model=config.models.llm_model,
        messages=[
            {"role": "system", "content": STRUCTURAL_PROMPT},
            {"role": "user", "content": query_text},
        ],
        temperature=0.2,
        max_tokens=200,
    )
    return response.choices[0].message.content or query_text


async def _expand_style(query_text: str) -> str:
    """Generate a style description from the query via async LLM."""
    config = load_config()
    client = get_async_openai_client()
    response = await client.chat.completions.create(
        model=config.models.llm_model,
        messages=[
            {"role": "system", "content": STYLE_PROMPT},
            {"role": "user", "content": query_text},
        ],
        temperature=0.2,
        max_tokens=200,
    )
    return response.choices[0].message.content or query_text


async def _extract_image_descriptions(
    image_bytes: bytes,
) -> QueryImageDescriptions:
    """Extract semantic, structural, and style descriptions from an image via async GPT-4o.

    All parameters (``image_detail``, ``temperature``, ``max_tokens``,
    ``max_image_size``) are read from ``config.yaml`` under the
    ``query_extraction`` section.
    """
    config = load_config()
    qe = config.query_extraction
    client = get_async_openai_client()

    # Resize to reduce token count and speed up vision processing
    small = _resize_image_bytes(image_bytes, max_size=qe.max_image_size)
    b64 = base64.b64encode(small).decode()
    data_uri = f"data:image/jpeg;base64,{b64}"

    response = await client.beta.chat.completions.parse(
        model=config.models.llm_model,
        messages=[
            {"role": "system", "content": QUERY_IMAGE_SYSTEM_PROMPT},
            {  # type: ignore[misc, list-item]
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this image."},
                    {"type": "image_url", "image_url": {"url": data_uri, "detail": qe.image_detail}},
                ],
            },
        ],
        response_format=QueryImageDescriptions,
        temperature=qe.temperature,
        max_tokens=qe.max_tokens,
    )

    descriptions = response.choices[0].message.parsed
    if descriptions is None:
        msg = "GPT-4o returned no parsed descriptions for query image"
        raise ValueError(msg)
    return descriptions


# ---------------------------------------------------------------------------
# Text search vectors
# ---------------------------------------------------------------------------


async def generate_query_vectors(
    query_text: str,
) -> dict[str, list[float]]:
    """Generate semantic, structural, and style query vectors from text.

    Runs LLM expansion and semantic embedding concurrently:
      - Semantic embedding starts immediately (uses raw query).
      - Structural and style LLM expansions run in parallel.
      - Structural and style embeddings run once expansions complete.

    Args:
        query_text: Free-text search query.

    Returns:
        Dict with keys ``semantic_vector``, ``structural_vector``,
        ``style_vector``.
    """
    config = load_config()
    dims = config.index.vector_dimensions

    # Phase 1: LLM expansions + semantic embedding in parallel
    semantic_embed_task = asyncio.create_task(
        embed_text(query_text, dimensions=dims.semantic),
    )
    structural_desc_task = asyncio.create_task(_expand_structural(query_text))
    style_desc_task = asyncio.create_task(_expand_style(query_text))

    structural_desc, style_desc = await asyncio.gather(
        structural_desc_task, style_desc_task,
    )

    # Phase 2: remaining embeddings + collect semantic (likely already done)
    structural_vec, style_vec, semantic_vec = await asyncio.gather(
        embed_text(structural_desc, dimensions=dims.structural),
        embed_text(style_desc, dimensions=dims.style),
        semantic_embed_task,
    )

    return {
        "semantic_vector": semantic_vec,
        "structural_vector": structural_vec,
        "style_vector": style_vec,
    }


# ---------------------------------------------------------------------------
# Image search vectors (with SHA-256 cache for deterministic rankings)
# ---------------------------------------------------------------------------

# Cache stores query vectors keyed by SHA-256 hash of the image bytes.
# This ensures the same image always produces identical rankings,
# eliminating non-determinism from GPT-4o description generation.
_IMAGE_QUERY_CACHE: dict[str, dict[str, list[float]]] = {}
_IMAGE_CACHE_MAX_SIZE = 64


def _image_hash(image_bytes: bytes) -> str:
    """Compute SHA-256 hex digest for cache key."""
    return hashlib.sha256(image_bytes).hexdigest()


async def generate_image_query_vectors(
    image_bytes: bytes,
) -> dict[str, list[float]]:
    """Generate all 4 query vectors from a query image.

    Results are cached by SHA-256 hash so the same image always returns
    identical vectors and deterministic rankings.

    Runs GPT-4o extraction and image embedding concurrently, then
    embeds all 3 text descriptions in parallel:

    1. GPT-4o extracts semantic, structural, and style descriptions
       (async) while Cohere Embed v4 image embedding runs in parallel.
    2. Each description is embedded into its corresponding vector space
       via text-embedding-3-large (3 calls in parallel).
    3. All 4 vectors enable multi-vector RRF search against the index.

    Args:
        image_bytes: Raw image bytes (JPEG or PNG).

    Returns:
        Dict with keys ``semantic_vector``, ``structural_vector``,
        ``style_vector``, ``image_vector``.

    Raises:
        ValueError: When GPT-4o returns no parsed output.
    """
    cache_key = _image_hash(image_bytes)
    if cache_key in _IMAGE_QUERY_CACHE:
        logger.info("Image query vectors served from cache", hash=cache_key[:12])
        return _IMAGE_QUERY_CACHE[cache_key]

    config = load_config()
    dims = config.index.vector_dimensions

    # Phase 1: GPT-4o extraction + image embedding in parallel
    descriptions, img_vec = await asyncio.gather(
        _extract_image_descriptions(image_bytes),
        embed_image(image_bytes=image_bytes),
    )

    logger.info(
        "Query image descriptions extracted",
        semantic_len=len(descriptions.semantic_description),
        structural_len=len(descriptions.structural_description),
        style_len=len(descriptions.style_description),
    )

    # Phase 2: 3 text embeddings in parallel
    sem_vec, struct_vec, style_vec = await asyncio.gather(
        embed_text(descriptions.semantic_description, dimensions=dims.semantic),
        embed_text(descriptions.structural_description, dimensions=dims.structural),
        embed_text(descriptions.style_description, dimensions=dims.style),
    )

    result = {
        "semantic_vector": sem_vec,
        "structural_vector": struct_vec,
        "style_vector": style_vec,
        "image_vector": img_vec,
    }

    # Evict oldest entry if cache is full
    if len(_IMAGE_QUERY_CACHE) >= _IMAGE_CACHE_MAX_SIZE:
        oldest = next(iter(_IMAGE_QUERY_CACHE))
        del _IMAGE_QUERY_CACHE[oldest]

    _IMAGE_QUERY_CACHE[cache_key] = result
    return result


def generate_query_vectors_sync(
    query_text: str,
) -> dict[str, list[float]]:
    """Synchronous wrapper for query vector generation."""
    return asyncio.run(generate_query_vectors(query_text))
