"""Unified retrieval pipeline — text-to-image and image-to-image search."""

from __future__ import annotations

import asyncio

import structlog

from ai_search.config import load_config
from ai_search.models import SearchMode, SearchResult
from ai_search.retrieval.query import generate_image_query_vectors, generate_query_vectors
from ai_search.retrieval.search import execute_vector_search

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


async def search(
    *,
    mode: SearchMode = SearchMode.TEXT,
    query_text: str | None = None,
    query_image_bytes: bytes | None = None,
    odata_filter: str | None = None,
    top: int | None = None,
) -> list[SearchResult]:
    """Unified search across text-to-image and image-to-image modes.

    Args:
        mode: ``TEXT`` for hybrid BM25 + multi-vector search,
              ``IMAGE`` for direct cosine-similarity on image embeddings.
        query_text: Free-text query (required for TEXT mode).
        query_image_bytes: Raw image bytes — JPEG or PNG (required for IMAGE mode).
        odata_filter: Optional OData filter expression (TEXT mode only).
        top: Maximum results to return.  Falls back to ``config.retrieval.top_k``.

    Returns:
        Ranked list of :class:`SearchResult` with ``search_score`` in [0, 1].

    Raises:
        ValueError: When required inputs for the selected mode are missing.
    """
    config = load_config()
    final_top = top or config.retrieval.top_k

    if mode == SearchMode.IMAGE:
        return await _search_by_image(query_image_bytes, final_top)

    return await _search_by_text(query_text, odata_filter, final_top)


# ---------------------------------------------------------------------------
# Text search path (multi-vector RRF)
# ---------------------------------------------------------------------------


async def _search_by_text(
    query_text: str | None,
    odata_filter: str | None,
    top: int,
) -> list[SearchResult]:
    """Execute text-to-image search via 3-vector RRF.

    Generates semantic, structural, and style vectors from the query
    text, then runs a pure multi-vector search against Azure AI Search.
    Azure fuses the three result sets via Reciprocal Rank Fusion (RRF).
    """
    if not query_text or not query_text.strip():
        msg = "query_text is required for TEXT search mode"
        raise ValueError(msg)

    query_vectors = await generate_query_vectors(query_text)

    search_docs = execute_vector_search(
        query_vectors=query_vectors,
        odata_filter=odata_filter,
        top=top,
    )

    results = _docs_to_results(search_docs)
    logger.info("Text search complete", result_count=len(results))
    return results


# ---------------------------------------------------------------------------
# Image search path (GPT-4o extraction + 4-vector RRF)
# ---------------------------------------------------------------------------


async def _search_by_image(
    query_image_bytes: bytes | None,
    top: int,
) -> list[SearchResult]:
    """Execute image-to-image search via GPT-4o extraction and 4-vector RRF.

    1. Embeds the query image via Cohere Embed v4 (image_vector).
    2. Extracts semantic, structural, and style descriptions via GPT-4o.
    3. Embeds each description into its corresponding text vector space.
    4. Runs a pure multi-vector search with all 4 vectors against Azure
       AI Search, which fuses results via RRF reranking.
    """
    if not query_image_bytes:
        msg = "query_image_bytes is required for IMAGE search mode"
        raise ValueError(msg)

    query_vectors = await generate_image_query_vectors(query_image_bytes)
    search_docs = execute_vector_search(
        query_vectors=query_vectors,
        top=top,
        search_mode=SearchMode.IMAGE,
    )

    results = _docs_to_results(search_docs)
    logger.info("Image search complete", result_count=len(results))
    return results


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _docs_to_results(docs: list[dict]) -> list[SearchResult]:
    """Convert raw search documents to :class:`SearchResult` models."""
    return [
        SearchResult(
            image_id=doc.get("image_id", ""),
            search_score=doc.get("search_score", 0.0),
            generation_prompt=doc.get("generation_prompt"),
            image_url=doc.get("image_url"),
            scene_type=doc.get("scene_type"),
            tags=doc.get("tags", []),
        )
        for doc in docs
    ]


# ---------------------------------------------------------------------------
# Legacy helpers (backward compatibility)
# ---------------------------------------------------------------------------


async def retrieve(
    query_text: str,
    odata_filter: str | None = None,
    top: int | None = None,
) -> list[SearchResult]:
    """Execute hybrid search retrieval pipeline.

    .. deprecated:: Use :func:`search` with ``mode=SearchMode.TEXT`` instead.
    """
    return await search(
        mode=SearchMode.TEXT,
        query_text=query_text,
        odata_filter=odata_filter,
        top=top,
    )


def retrieve_sync(
    query_text: str,
    odata_filter: str | None = None,
    top: int | None = None,
) -> list[SearchResult]:
    """Synchronous wrapper for the retrieval pipeline."""
    return asyncio.run(retrieve(query_text, odata_filter, top))
