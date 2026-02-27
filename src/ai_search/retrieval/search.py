"""Hybrid search execution with configurable multi-vector weights."""

from __future__ import annotations

from typing import Any

import structlog
from azure.search.documents.models import VectorizedQuery

from ai_search.clients import get_search_client
from ai_search.config import load_config
from ai_search.retrieval.relevance import (
    Confidence,
    RelevanceResult,
    filter_by_relevance,
)

logger = structlog.get_logger(__name__)

# Fields to retrieve from the index
SELECT_FIELDS = [
    "image_id",
    "generation_prompt",
    "scene_type",
    "tags",
    "narrative_type",
    "emotional_polarity",
    "low_light_score",
    "character_count",
    "extraction_json",
    "metadata_json",
]


def execute_hybrid_search(
    query_text: str,
    query_vectors: dict[str, list[float]],
    odata_filter: str | None = None,
    top: int | None = None,
    min_confidence: Confidence | None = None,
) -> tuple[list[dict[str, Any]], RelevanceResult | None]:
    """Execute hybrid search against Azure AI Search.

    Combines BM25 text search with weighted multi-vector queries.
    Vector weights are config weights × 10 to preserve ratio vs BM25's implicit 1.0.

    Args:
        query_text: Free-text search query for BM25 matching.
        query_vectors: Mapping of vector field names to query vectors.
        odata_filter: Optional OData filter expression.
        top: Override for number of results to return.
        min_confidence: If set, apply relevance filtering and discard results
            below this confidence tier.  ``None`` skips filtering.

    Returns:
        Tuple of (documents, relevance).  ``relevance`` is ``None`` when
        ``min_confidence`` is not set.
    """
    config = load_config()
    client = get_search_client()
    weights = config.search
    retrieval = config.retrieval
    search_top = top or retrieval.top_k

    # Build vector queries with config weight × 10
    vector_queries = []

    if "semantic_vector" in query_vectors:
        vector_queries.append(
            VectorizedQuery(
                vector=query_vectors["semantic_vector"],
                k_nearest_neighbors=retrieval.k_nearest,
                fields="semantic_vector",
                weight=weights.semantic_weight * 10,
            )
        )

    if "structural_vector" in query_vectors:
        vector_queries.append(
            VectorizedQuery(
                vector=query_vectors["structural_vector"],
                k_nearest_neighbors=retrieval.k_nearest,
                fields="structural_vector",
                weight=weights.structural_weight * 10,
            )
        )

    if "style_vector" in query_vectors:
        vector_queries.append(
            VectorizedQuery(
                vector=query_vectors["style_vector"],
                k_nearest_neighbors=retrieval.k_nearest,
                fields="style_vector",
                weight=weights.style_weight * 10,
            )
        )

    if "image_vector" in query_vectors:
        vector_queries.append(
            VectorizedQuery(
                vector=query_vectors["image_vector"],
                k_nearest_neighbors=retrieval.k_nearest,
                fields="image_vector",
                weight=weights.image_weight * 10,
            )
        )

    results = client.search(
        search_text=query_text,
        vector_queries=vector_queries,  # type: ignore[arg-type]
        filter=odata_filter,
        select=SELECT_FIELDS,
        top=search_top,
    )

    documents = []
    for result in results:
        doc = dict(result)
        doc["search_score"] = result.get("@search.score", 0.0)
        documents.append(doc)

    logger.info("Hybrid search complete", result_count=len(documents), top=search_top)

    # Optional relevance filtering
    if min_confidence is not None:
        filtered, relevance = filter_by_relevance(
            documents, score_key="search_score", min_confidence=min_confidence,
        )
        return filtered, relevance

    return documents, None
