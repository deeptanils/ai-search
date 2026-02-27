"""Single-stage retrieval pipeline — hybrid search with RRF fusion."""

from __future__ import annotations

import asyncio

import structlog

from ai_search.config import load_config
from ai_search.models import SearchResult
from ai_search.retrieval.query import generate_query_vectors
from ai_search.retrieval.search import execute_hybrid_search

logger = structlog.get_logger(__name__)


async def retrieve(
    query_text: str,
    odata_filter: str | None = None,
    top: int | None = None,
) -> list[SearchResult]:
    """Execute hybrid search retrieval pipeline.

    Uses Azure AI Search hybrid query with BM25 + multi-vector RRF fusion.
    """
    config = load_config()
    final_top = top or config.retrieval.top_k

    # Generate query embeddings
    query_vectors = await generate_query_vectors(query_text)

    # Hybrid search (BM25 + vector RRF)
    search_results = execute_hybrid_search(
        query_text=query_text,
        query_vectors=query_vectors,
        odata_filter=odata_filter,
        top=final_top,
    )

    if not search_results:
        logger.info("No results from hybrid search")
        return []

    # Build results
    results = [
        SearchResult(
            image_id=doc.get("image_id", ""),
            search_score=doc.get("search_score", 0.0),
            generation_prompt=doc.get("generation_prompt"),
            scene_type=doc.get("scene_type"),
            tags=doc.get("tags", []),
        )
        for doc in search_results
    ]

    logger.info("Retrieval complete", result_count=len(results))
    return results


def retrieve_sync(
    query_text: str,
    odata_filter: str | None = None,
    top: int | None = None,
) -> list[SearchResult]:
    """Synchronous wrapper for the retrieval pipeline."""
    return asyncio.run(retrieve(query_text, odata_filter, top))
