"""Azure AI Search index schema definition with HNSW vector configuration."""

from __future__ import annotations

import structlog
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    ScoringProfile,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    TextWeights,
    VectorSearch,
    VectorSearchProfile,
)

from ai_search.clients import get_search_index_client
from ai_search.config import load_config

logger = structlog.get_logger(__name__)

ALGORITHM_NAME = "hnsw-cosine"
PROFILE_NAME = "hnsw-cosine-profile"


def _build_vector_field(
    name: str,
    dimensions: int,
    profile: str = PROFILE_NAME,
) -> SearchField:
    """Build a vector search field definition."""
    return SearchField(
        name=name,
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=dimensions,
        vector_search_profile_name=profile,
    )


def build_index_schema() -> SearchIndex:
    """Build the complete Azure AI Search index definition."""
    config = load_config()
    idx = config.index
    dims = idx.vector_dimensions
    hnsw = idx.hnsw

    # HNSW algorithm configuration
    hnsw_algo = HnswAlgorithmConfiguration(
        name=ALGORITHM_NAME,
        parameters=HnswParameters(
            m=hnsw.m,
            ef_construction=hnsw.ef_construction,
            ef_search=hnsw.ef_search,
            metric="cosine",
        ),
    )

    vector_search = VectorSearch(
        algorithms=[hnsw_algo],
        profiles=[
            VectorSearchProfile(
                name=PROFILE_NAME,
                algorithm_configuration_name=ALGORITHM_NAME,
            ),
        ],
    )

    # Primitive fields
    fields: list[SearchField] = [
        SimpleField(name="image_id", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchableField(name="generation_prompt", type=SearchFieldDataType.String),
        SimpleField(name="scene_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="time_of_day", type=SearchFieldDataType.String, filterable=True),
        SimpleField(
            name="lighting_condition", type=SearchFieldDataType.String, filterable=True, facetable=True
        ),
        SimpleField(name="primary_subject", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="artistic_style", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SearchField(
            name="tags",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            searchable=True,
            filterable=True,
            facetable=True,
        ),
        SimpleField(name="narrative_theme", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="narrative_type", type=SearchFieldDataType.String, filterable=True),
        SimpleField(
            name="emotional_polarity", type=SearchFieldDataType.Double, filterable=True, sortable=True
        ),
        SimpleField(name="low_light_score", type=SearchFieldDataType.Double, filterable=True),
        SimpleField(
            name="character_count", type=SearchFieldDataType.Int32, filterable=True, sortable=True
        ),
        SimpleField(name="metadata_json", type=SearchFieldDataType.String),
        SimpleField(name="extraction_json", type=SearchFieldDataType.String),
    ]

    # Primary vector fields
    fields.extend([
        _build_vector_field("semantic_vector", dims.semantic),
        _build_vector_field("structural_vector", dims.structural),
        _build_vector_field("style_vector", dims.style),
        _build_vector_field("image_vector", dims.image),
    ])

    # Text-boost scoring profile
    text_boost_profile = ScoringProfile(
        name="text-boost",
        text_weights=TextWeights(
            weights={
                "generation_prompt": 3.0,
                "tags": 2.0,
            }
        ),
    )

    return SearchIndex(
        name=idx.name,
        fields=fields,
        vector_search=vector_search,
        scoring_profiles=[text_boost_profile],
        default_scoring_profile="text-boost",
    )


def create_or_update_index() -> SearchIndex:
    """Create or update the search index in Azure AI Search."""
    index = build_index_schema()
    client = get_search_index_client()

    result = client.create_or_update_index(index)
    logger.info("Index created/updated", index_name=result.name)
    return result
