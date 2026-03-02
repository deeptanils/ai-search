"""Multi-vector search execution with RRF reranking."""

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

from ai_search.models import SearchMode

logger = structlog.get_logger(__name__)

# Fields to retrieve from the index
SELECT_FIELDS = [
    "image_id",
    "generation_prompt",
    "image_url",
    "scene_type",
    "character_action",
    "weapons_props",
    "location_name",
    "episode_name",
    "tags",
    "narrative_type",
    "emotional_polarity",
    "low_light_score",
    "character_count",
    "extraction_json",
    "metadata_json",
]


def execute_vector_search(
    query_vectors: dict[str, list[float]],
    odata_filter: str | None = None,
    top: int | None = None,
    search_mode: SearchMode = SearchMode.TEXT,
) -> list[dict[str, Any]]:
    """Execute pure multi-vector search with RRF reranking.

    Passes multiple vector queries to Azure AI Search, which fuses
    them via Reciprocal Rank Fusion (RRF) internally.  No BM25
    keyword search is involved.

    Args:
        query_vectors: Mapping of vector field names to query vectors.
            Accepted keys: ``semantic_vector``, ``structural_vector``,
            ``style_vector``, ``image_vector``.
        odata_filter: Optional OData filter expression.
        top: Override for number of results to return.
        search_mode: Determines which weight profile to use.
            ``IMAGE`` mode boosts the image_vector weight.

    Returns:
        Documents sorted by RRF score, each with a ``search_score``
        normalized to [0, 1].
    """
    config = load_config()
    client = get_search_client()
    weights = config.image_search if search_mode == SearchMode.IMAGE else config.search
    retrieval = config.retrieval
    search_top = top or retrieval.top_k

    weight_map = {
        "semantic_vector": weights.semantic_weight * 10,
        "structural_vector": weights.structural_weight * 10,
        "style_vector": weights.style_weight * 10,
        "image_vector": weights.image_weight * 10,
    }

    vector_queries = [
        VectorizedQuery(
            vector=vector,
            k_nearest_neighbors=retrieval.k_nearest,
            fields=field_name,
            weight=weight_map.get(field_name, 1.0),
        )
        for field_name, vector in query_vectors.items()
    ]

    results = client.search(
        search_text=None,
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

    _normalize_scores(documents)

    logger.info(
        "Vector search complete",
        result_count=len(documents),
        vector_fields=list(query_vectors.keys()),
        top=search_top,
    )
    return documents


def execute_hybrid_search(
    query_text: str,
    query_vectors: dict[str, list[float]],
    odata_filter: str | None = None,
    top: int | None = None,
    min_confidence: Confidence | None = None,
) -> tuple[list[dict[str, Any]], RelevanceResult | None]:
    """Execute hybrid BM25 + multi-vector search.

    .. deprecated::
        Use :func:`execute_vector_search` instead.  Retained for
        backward compatibility with existing callers.
    """
    config = load_config()
    client = get_search_client()
    weights = config.search
    retrieval = config.retrieval
    search_top = top or retrieval.top_k

    weight_map = {
        "semantic_vector": weights.semantic_weight * 10,
        "structural_vector": weights.structural_weight * 10,
        "style_vector": weights.style_weight * 10,
        "image_vector": weights.image_weight * 10,
    }

    vector_queries = [
        VectorizedQuery(
            vector=vector,
            k_nearest_neighbors=retrieval.k_nearest,
            fields=field_name,
            weight=weight_map.get(field_name, 1.0),
        )
        for field_name, vector in query_vectors.items()
    ]

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

    _normalize_scores(documents)

    logger.info("Hybrid search complete", result_count=len(documents), top=search_top)

    if min_confidence is not None:
        filtered, relevance = filter_by_relevance(
            documents, score_key="search_score", min_confidence=min_confidence,
        )
        return filtered, relevance

    return documents, None


def _normalize_scores(documents: list[dict[str, Any]]) -> None:
    """Normalize search_score values to 0-1 range in place.

    Uses min-max normalization so the top result scores 1.0 and others
    are scaled proportionally.  When all scores are identical the
    function assigns 1.0 to every document.
    """
    if not documents:
        return

    scores = [doc["search_score"] for doc in documents]
    max_score = max(scores)
    min_score = min(scores)
    score_range = max_score - min_score

    for doc in documents:
        if score_range > 0:
            doc["search_score"] = (doc["search_score"] - min_score) / score_range
        else:
            doc["search_score"] = 1.0



