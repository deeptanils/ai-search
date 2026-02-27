---
title: Azure AI Search Python SDK Research for Multi-Vector Search Index
description: Comprehensive research on azure-search-documents SDK capabilities for building a multi-vector search index with hybrid retrieval
author: researcher-subagent
ms.date: 2026-02-26
ms.topic: reference
keywords:
  - azure ai search
  - python sdk
  - vector search
  - hybrid search
  - multi-vector index
  - HNSW
estimated_reading_time: 25
---

## Overview

This document captures research findings on the `azure-search-documents` Python SDK for building a multi-vector search index aligned with the candidate generation pipeline defined in the project requirements. All code examples target the latest stable SDK API surface.

## 1. Package Installation and Client Initialization

### 1.1 Package Details

The `azure-search-documents` package is the official Python SDK for Azure AI Search (formerly Azure Cognitive Search).

```bash
pip install azure-search-documents
```

When using UV as the dependency manager (per project requirements):

```bash
uv add azure-search-documents
```

The SDK consists of two primary clients:

* `SearchIndexClient` for index management (create, update, delete indexes)
* `SearchClient` for document operations (upload, search, suggest, autocomplete)

### 1.2 Authentication with API Key

```python
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient

endpoint = os.environ["AZURE_AI_SEARCH_ENDPOINT"]
api_key = os.environ["AZURE_AI_SEARCH_API_KEY"]
index_name = os.environ["AZURE_AI_SEARCH_INDEX_NAME"]

credential = AzureKeyCredential(api_key)

# For index management
index_client = SearchIndexClient(endpoint=endpoint, credential=credential)

# For document operations
search_client = SearchClient(
    endpoint=endpoint,
    index_name=index_name,
    credential=credential,
)
```

### 1.3 Authentication with Azure Identity (Alternative)

```python
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
```

> [!NOTE]
> The project requirements specify API key authentication stored in `.env`. Azure Identity is documented here as a production-grade alternative for managed identity scenarios.

## 2. Index Creation with Multiple Vector Fields

### 2.1 Complete Index Schema Definition

The SDK uses `SearchIndex`, `SearchField`, `SearchableField`, `SimpleField`, and `SearchFieldDataType` to define index schemas. Vector fields use `SearchField` with `vector_search_dimensions` and `vector_search_profile_name`.

```python
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SearchableField,
    SimpleField,
    ComplexField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)


fields = [
    # --- Primitive fields ---
    SimpleField(
        name="image_id",
        type=SearchFieldDataType.String,
        key=True,
        filterable=True,
    ),
    SearchableField(
        name="generation_prompt",
        type=SearchFieldDataType.String,
    ),
    SimpleField(
        name="scene_type",
        type=SearchFieldDataType.String,
        filterable=True,
    ),
    SimpleField(
        name="lighting_condition",
        type=SearchFieldDataType.String,
        filterable=True,
    ),
    SearchableField(
        name="tags",
        type=SearchFieldDataType.Collection(SearchFieldDataType.String),
        filterable=True,
        facetable=True,
    ),
    SimpleField(
        name="emotional_polarity",
        type=SearchFieldDataType.Double,
        filterable=True,
        sortable=True,
    ),
    SimpleField(
        name="low_light_score",
        type=SearchFieldDataType.Double,
        filterable=True,
    ),

    # --- Vector fields ---
    SearchField(
        name="semantic_vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=3072,
        vector_search_profile_name="hnsw-cosine-profile",
    ),
    SearchField(
        name="structural_vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=1024,
        vector_search_profile_name="hnsw-cosine-profile",
    ),
    SearchField(
        name="style_vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=512,
        vector_search_profile_name="hnsw-cosine-profile",
    ),
]
```

### 2.2 Vector Field Requirements

Each vector field requires three properties:

* `type`: Must be `Collection(Edm.Single)` (mapped via `SearchFieldDataType.Collection(SearchFieldDataType.Single)`)
* `vector_search_dimensions`: Integer matching the embedding model output dimensionality
* `vector_search_profile_name`: References a named `VectorSearchProfile` configured on the index

### 2.3 Handling Nested/Complex Types for Character Vectors

Azure AI Search supports `ComplexField` for nested structures. However, vector search on nested complex type fields has specific constraints.

#### Option A: Flatten Character Vectors (Recommended for Search)

Azure AI Search does not support vector search directly on fields inside `Collection(Edm.ComplexType)`. The recommended approach is to flatten character vectors into top-level fields or use a single aggregated character vector.

```python
# Flatten: store the primary character's semantic vector at top level
SearchField(
    name="primary_character_semantic_vector",
    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
    searchable=True,
    vector_search_dimensions=3072,
    vector_search_profile_name="hnsw-cosine-profile",
),
```

#### Option B: Complex Type for Metadata, Separate Vector Fields

Store character metadata in a complex type but keep vectors at the top level.

```python
# Character metadata as complex type (non-vector fields)
ComplexField(
    name="character_vectors",
    collection=True,
    fields=[
        SimpleField(name="character_id", type=SearchFieldDataType.String),
        # Non-searchable storage of vectors as strings or separate index
    ],
),

# Aggregated character semantic vector at top level for search
SearchField(
    name="character_semantic_vector_aggregated",
    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
    searchable=True,
    vector_search_dimensions=3072,
    vector_search_profile_name="hnsw-cosine-profile",
),
```

#### Option C: Separate Character Index

For scenarios where per-character vector search is critical, create a separate index for characters with a foreign key back to the parent image.

```python
character_fields = [
    SimpleField(name="character_doc_id", type=SearchFieldDataType.String, key=True),
    SimpleField(name="image_id", type=SearchFieldDataType.String, filterable=True),
    SimpleField(name="character_id", type=SearchFieldDataType.String, filterable=True),
    SearchField(
        name="semantic",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=3072,
        vector_search_profile_name="hnsw-cosine-profile",
    ),
    SearchField(
        name="emotion",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=512,
        vector_search_profile_name="hnsw-cosine-profile",
    ),
    SearchField(
        name="pose",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=512,
        vector_search_profile_name="hnsw-cosine-profile",
    ),
]
```

> [!IMPORTANT]
> Azure AI Search does not support vector search on fields nested inside `Collection(Edm.ComplexType)`. Vector fields must be top-level or inside a non-collection complex type. For the requirements specifying `character_vectors.semantic` as vector-search-enabled, use Option A (flattened) or Option C (separate index).

## 3. Vector Search Configuration

### 3.1 HNSW Algorithm Configuration

```python
from azure.search.documents.indexes.models import (
    VectorSearch,
    HnswAlgorithmConfiguration,
    HnswParameters,
    VectorSearchProfile,
    VectorSearchAlgorithmMetric,
)

vector_search = VectorSearch(
    algorithms=[
        HnswAlgorithmConfiguration(
            name="hnsw-cosine",
            parameters=HnswParameters(
                m=4,           # Number of bi-directional links per node
                ef_construction=400,  # Size of dynamic list during indexing
                ef_search=500,        # Size of dynamic list during search
                metric=VectorSearchAlgorithmMetric.COSINE,
            ),
        ),
    ],
    profiles=[
        VectorSearchProfile(
            name="hnsw-cosine-profile",
            algorithm_configuration_name="hnsw-cosine",
        ),
    ],
)
```

### 3.2 HNSW Parameter Guidance

| Parameter        | Default | Recommended for 10M+ docs | Description                                       |
|------------------|---------|---------------------------|---------------------------------------------------|
| `m`              | 4       | 4-10                      | Higher values improve recall at cost of memory     |
| `ef_construction`| 400     | 400-1000                  | Higher values improve index quality, slower builds |
| `ef_search`      | 500     | 500-1000                  | Higher values improve recall, increase latency     |

> [!TIP]
> For 10M+ documents, start with `m=4, ef_construction=400, ef_search=500`. Increase `ef_search` if recall is insufficient. Azure AI Search automatically manages shard distribution.

### 3.3 Exhaustive KNN (Alternative)

For exact nearest-neighbor search (useful for evaluation/testing):

```python
from azure.search.documents.indexes.models import ExhaustiveKnnAlgorithmConfiguration

algorithms=[
    ExhaustiveKnnAlgorithmConfiguration(
        name="exhaustive-cosine",
        parameters={"metric": "cosine"},
    ),
]
```

### 3.4 Assembling the Complete Index

```python
index = SearchIndex(
    name=index_name,
    fields=fields,
    vector_search=vector_search,
)

# Create or update the index
result = index_client.create_or_update_index(index)
print(f"Index '{result.name}' created/updated successfully")
```

## 4. Hybrid Search: BM25 + Multiple Vector Queries

### 4.1 VectorizedQuery for Multi-Vector Search

The `search()` method accepts a `vector_queries` parameter that takes a list of `VectorizedQuery` objects. Each query targets a specific vector field.

```python
from azure.search.documents.models import VectorizedQuery

# Assume these are pre-computed query embeddings
query_semantic_vector = [...]    # 3072 dims
query_structural_vector = [...]  # 1024 dims
query_style_vector = [...]       # 512 dims

results = search_client.search(
    search_text="cinematic night scene with dramatic lighting",  # BM25 keyword search
    vector_queries=[
        VectorizedQuery(
            vector=query_semantic_vector,
            k_nearest_neighbors=50,
            fields="semantic_vector",
        ),
        VectorizedQuery(
            vector=query_structural_vector,
            k_nearest_neighbors=50,
            fields="structural_vector",
        ),
        VectorizedQuery(
            vector=query_style_vector,
            k_nearest_neighbors=50,
            fields="style_vector",
        ),
    ],
    top=20,
)

for result in results:
    print(f"Score: {result['@search.score']}, Image: {result['image_id']}")
```

### 4.2 Key Parameters for VectorizedQuery

| Parameter              | Type       | Description                                      |
|------------------------|------------|--------------------------------------------------|
| `vector`               | list[float]| The query embedding vector                       |
| `k_nearest_neighbors`  | int        | Number of nearest neighbors to retrieve per field|
| `fields`               | str        | Comma-separated target vector field names        |
| `exhaustive`           | bool       | Use exhaustive KNN instead of HNSW (optional)    |
| `weight`               | float      | Relative weight for this vector query (optional) |

### 4.3 Hybrid Search Behavior

When both `search_text` and `vector_queries` are provided, Azure AI Search performs:

1. BM25 text search on searchable fields
2. Each vector query independently against its target field
3. Results are fused using Reciprocal Rank Fusion (RRF) by default

RRF combines ranked lists from each retrieval method into a single ranked result set without requiring score normalization.

### 4.4 Search Modes

```python
results = search_client.search(
    search_text="night scene",
    search_mode="all",       # "any" (default) or "all" for keyword matching
    query_type="simple",     # "simple", "full" (Lucene), or "semantic"
    vector_queries=[...],
    top=20,
)
```

## 5. Scoring Profiles

### 5.1 Scoring Profile for Weighted Field Boosting

Scoring profiles in Azure AI Search apply to text/keyword search scoring. They do not directly weight vector similarity scores. Vector query weighting uses the `weight` parameter on `VectorizedQuery`.

```python
from azure.search.documents.indexes.models import (
    ScoringProfile,
    TextWeights,
)

scoring_profiles = [
    ScoringProfile(
        name="weighted-text-profile",
        text_weights=TextWeights(
            weights={
                "generation_prompt": 2.0,
                "tags": 1.5,
            },
        ),
    ),
]

index = SearchIndex(
    name=index_name,
    fields=fields,
    vector_search=vector_search,
    scoring_profiles=scoring_profiles,
    default_scoring_profile="weighted-text-profile",
)
```

### 5.2 Weighted Vector Queries

To achieve the weighted scoring formula from the requirements:

`FinalScore = w1 * semantic + w2 * structural + w3 * style + w4 * keyword`

Use the `weight` parameter on each `VectorizedQuery` and rely on RRF for fusion:

```python
from azure.search.documents.models import VectorizedQuery

# Weights from config.yaml
semantic_weight = 0.5
structural_weight = 0.2
style_weight = 0.2

results = search_client.search(
    search_text="dramatic cinematic lighting",
    vector_queries=[
        VectorizedQuery(
            vector=query_semantic_vector,
            k_nearest_neighbors=50,
            fields="semantic_vector",
            weight=semantic_weight,
        ),
        VectorizedQuery(
            vector=query_structural_vector,
            k_nearest_neighbors=50,
            fields="structural_vector",
            weight=structural_weight,
        ),
        VectorizedQuery(
            vector=query_style_vector,
            k_nearest_neighbors=50,
            fields="style_vector",
            weight=style_weight,
        ),
    ],
    top=20,
)
```

> [!NOTE]
> The `weight` parameter on `VectorizedQuery` adjusts the contribution of each vector subsearch within the RRF fusion. This is the primary mechanism for achieving weighted multi-vector retrieval. Keyword (BM25) weight is indirectly controlled through scoring profiles and the RRF fusion process.

### 5.3 Scoring Profile with Functions

For boost functions based on field values (freshness, magnitude, distance, tag):

```python
from azure.search.documents.indexes.models import (
    ScoringProfile,
    ScoringFunction,
    MagnitudeScoringFunction,
    MagnitudeScoringParameters,
    ScoringFunctionAggregation,
)

scoring_profiles = [
    ScoringProfile(
        name="boost-low-light",
        functions=[
            MagnitudeScoringFunction(
                field_name="low_light_score",
                boost=2.0,
                parameters=MagnitudeScoringParameters(
                    boosting_range_start=0.0,
                    boosting_range_end=1.0,
                ),
                interpolation="linear",
            ),
        ],
        function_aggregation=ScoringFunctionAggregation.SUM,
    ),
]
```

## 6. Batch Indexing

### 6.1 IndexDocumentsBatch for Upload

```python
from azure.search.documents import SearchClient
from azure.search.documents.models import IndexDocumentsBatch

batch = IndexDocumentsBatch()

# Add documents to the batch
for doc in documents:
    batch.add_upload_actions(doc)

# Upload
result = search_client.index_documents(batch)
print(f"Uploaded {len(result.results)} documents")
```

### 6.2 Batch Actions

The `IndexDocumentsBatch` supports four action types:

| Method                     | Behavior                                            |
|----------------------------|-----------------------------------------------------|
| `add_upload_actions`       | Inserts or replaces the document                    |
| `add_merge_actions`        | Merges fields into existing document                |
| `add_merge_or_upload_actions` | Merges if exists, uploads if new                 |
| `add_delete_actions`       | Deletes the document by key                         |

### 6.3 Batch Size Limits and Strategies for 10M+ Documents

| Constraint                     | Limit                                           |
|--------------------------------|-------------------------------------------------|
| Max documents per batch        | 1,000 documents                                 |
| Max batch payload size         | 16 MB                                           |
| Max document size              | 16 MB                                           |
| Max vector dimensions per field| 3,072                                           |

For 10M+ documents, the recommended approach:

```python
import time
from azure.core.exceptions import HttpResponseError

def upload_documents_in_batches(
    search_client: SearchClient,
    documents: list,
    batch_size: int = 500,
    max_retries: int = 3,
    retry_delay: float = 2.0,
):
    """Upload documents in batches with retry logic."""
    total = len(documents)
    succeeded = 0
    failed = 0

    for i in range(0, total, batch_size):
        batch_docs = documents[i : i + batch_size]
        batch = IndexDocumentsBatch()
        batch.add_upload_actions(batch_docs)

        for attempt in range(max_retries):
            try:
                result = search_client.index_documents(batch)
                batch_succeeded = sum(
                    1 for r in result.results if r.succeeded
                )
                batch_failed = sum(
                    1 for r in result.results if not r.succeeded
                )
                succeeded += batch_succeeded
                failed += batch_failed
                break
            except HttpResponseError as e:
                if e.status_code == 429:  # Too Many Requests
                    wait = retry_delay * (2 ** attempt)
                    time.sleep(wait)
                elif e.status_code == 503:  # Service Unavailable
                    wait = retry_delay * (2 ** attempt)
                    time.sleep(wait)
                else:
                    raise

        if (i // batch_size) % 100 == 0:
            print(f"Progress: {i + len(batch_docs)}/{total}")

    return succeeded, failed
```

### 6.4 Optimization Strategies for Large-Scale Indexing

* Reduce batch size to 500 when documents contain multiple high-dimensional vectors (the combined payload approaches the 16 MB limit faster).
* Use parallel workers with separate `SearchClient` instances (the client is thread-safe).
* Implement exponential backoff for 429 and 503 responses.
* Use `add_merge_or_upload_actions` for incremental re-indexing.
* Monitor indexer throughput using Azure Monitor metrics.
* Consider partitioning the upload across multiple processes using `concurrent.futures`:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def upload_batch(search_client, batch_docs):
    batch = IndexDocumentsBatch()
    batch.add_upload_actions(batch_docs)
    return search_client.index_documents(batch)

# Split documents into chunks
chunks = [documents[i:i+500] for i in range(0, len(documents), 500)]

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {
        executor.submit(upload_batch, search_client, chunk): idx
        for idx, chunk in enumerate(chunks)
    }
    for future in as_completed(futures):
        result = future.result()
        print(f"Batch {futures[future]}: {len(result.results)} processed")
```

### 6.5 Document Structure for Upload

Each document is a dictionary that matches the index schema:

```python
document = {
    "image_id": "img_001",
    "generation_prompt": "A cinematic night scene with dramatic lighting",
    "scene_type": "cinematic",
    "lighting_condition": "low-key",
    "tags": ["night", "dramatic", "cinematic", "urban"],
    "emotional_polarity": 0.65,
    "low_light_score": 0.82,
    "semantic_vector": [0.012, -0.034, ...],      # 3072 floats
    "structural_vector": [0.056, 0.078, ...],     # 1024 floats
    "style_vector": [-0.023, 0.045, ...],         # 512 floats
    "primary_character_semantic_vector": [0.011, ...],  # 3072 floats (if flattened)
}
```

## 7. Faceting and Filtering with Vector Search

### 7.1 Filters

Filters are applied as OData expressions and execute before vector search, reducing the candidate set:

```python
results = search_client.search(
    search_text=None,
    vector_queries=[
        VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=50,
            fields="semantic_vector",
        ),
    ],
    filter="scene_type eq 'cinematic' and low_light_score gt 0.5",
    top=20,
)
```

#### Common Filter Expressions

| Expression                                                 | Purpose                             |
|------------------------------------------------------------|-------------------------------------|
| `scene_type eq 'cinematic'`                                | Exact match                         |
| `low_light_score gt 0.5`                                   | Numeric comparison                  |
| `emotional_polarity ge 0.3 and emotional_polarity le 0.8`  | Range filter                        |
| `tags/any(t: t eq 'night')`                                | Collection contains value           |
| `lighting_condition ne 'harsh'`                            | Not equal                           |
| `search.in(scene_type, 'cinematic,documentary,surreal')`   | Multi-value match                   |

### 7.2 Pre-filter vs. Post-filter

Azure AI Search applies filters as a pre-filter to vector search by default. This means the HNSW graph traversal operates only on the filtered subset. For large datasets with restrictive filters, this improves performance and ensures the `k` nearest neighbors come from the filtered set.

To use post-filtering, set `filter_mode`:

```python
from azure.search.documents.models import VectorFilterMode

results = search_client.search(
    vector_queries=[
        VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=50,
            fields="semantic_vector",
            filter_mode=VectorFilterMode.POST_FILTER,
        ),
    ],
    filter="scene_type eq 'cinematic'",
)
```

> [!TIP]
> Use pre-filtering (the default) for most scenarios. Post-filtering may return fewer than `k` results because it applies filters after the ANN search.

### 7.3 Facets

Facets return aggregated counts for structured fields alongside search results:

```python
results = search_client.search(
    search_text="dramatic lighting",
    vector_queries=[
        VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=50,
            fields="semantic_vector",
        ),
    ],
    facets=[
        "scene_type,count:10",
        "lighting_condition,count:10",
        "tags,count:20",
    ],
    top=20,
)

# Access facet results
for facet_name, facet_values in results.get_facets().items():
    print(f"\n{facet_name}:")
    for fv in facet_values:
        print(f"  {fv['value']}: {fv['count']}")
```

Facet expressions support:

| Syntax                          | Description                     |
|---------------------------------|---------------------------------|
| `field_name`                    | Default faceting                |
| `field_name,count:N`           | Limit number of facet values    |
| `field_name,sort:count`        | Sort by count (descending)      |
| `field_name,sort:value`        | Sort alphabetically             |
| `field_name,values:a\|b\|c`    | Specific facet values only      |
| `field_name,interval:N`       | Numeric interval bucketing      |

## 8. Re-ranking Options

### 8.1 Semantic Ranker (Built-in)

Azure AI Search includes a built-in semantic ranker that uses Microsoft-hosted language models to re-rank results based on semantic understanding.

```python
from azure.search.documents.indexes.models import (
    SemanticConfiguration,
    SemanticSearch,
    SemanticPrioritizedFields,
    SemanticField,
)

semantic_config = SemanticConfiguration(
    name="semantic-config",
    prioritized_fields=SemanticPrioritizedFields(
        content_fields=[
            SemanticField(field_name="generation_prompt"),
        ],
    ),
)

semantic_search = SemanticSearch(configurations=[semantic_config])

index = SearchIndex(
    name=index_name,
    fields=fields,
    vector_search=vector_search,
    semantic_search=semantic_search,
)
```

Querying with semantic re-ranking:

```python
results = search_client.search(
    search_text="cinematic night scene",
    vector_queries=[
        VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=50,
            fields="semantic_vector",
        ),
    ],
    query_type="semantic",
    semantic_configuration_name="semantic-config",
    top=20,
)

for result in results:
    print(
        f"Score: {result['@search.score']}, "
        f"Reranker: {result['@search.reranker_score']}, "
        f"Image: {result['image_id']}"
    )
```

### 8.2 Custom Re-ranking (Application-Level)

For the requirements specifying re-ranking by emotional alignment, narrative consistency, object overlap, and low-light compatibility, implement a two-stage retrieval:

```python
def search_and_rerank(
    search_client: SearchClient,
    query_text: str,
    query_vectors: dict,
    rerank_fn,
    initial_k: int = 100,
    final_k: int = 20,
):
    """Two-stage retrieval: broad recall, then custom re-rank."""
    # Stage 1: Broad retrieval
    results = search_client.search(
        search_text=query_text,
        vector_queries=[
            VectorizedQuery(
                vector=query_vectors["semantic"],
                k_nearest_neighbors=initial_k,
                fields="semantic_vector",
                weight=0.5,
            ),
            VectorizedQuery(
                vector=query_vectors["structural"],
                k_nearest_neighbors=initial_k,
                fields="structural_vector",
                weight=0.2,
            ),
            VectorizedQuery(
                vector=query_vectors["style"],
                k_nearest_neighbors=initial_k,
                fields="style_vector",
                weight=0.2,
            ),
        ],
        top=initial_k,
        select=[
            "image_id", "generation_prompt", "scene_type",
            "emotional_polarity", "low_light_score", "tags",
        ],
    )

    candidates = list(results)

    # Stage 2: Custom re-ranking
    reranked = rerank_fn(candidates, query_text, query_vectors)

    return reranked[:final_k]
```

### 8.3 LLM-based Re-ranking (Future Enhancement)

For the "online re-ranking with LLM reasoning" mentioned in the requirements, pass the candidate set to an LLM via Azure AI Foundry:

```python
def llm_rerank(candidates, query_context):
    """Re-rank candidates using an LLM for reasoning."""
    # Format candidates for the LLM prompt
    candidate_summaries = "\n".join(
        f"- ID: {c['image_id']}, Scene: {c['scene_type']}, "
        f"Emotion: {c['emotional_polarity']}"
        for c in candidates
    )

    prompt = f"""Given the query context: {query_context}
    
Rank these candidates by relevance. Consider emotional alignment,
narrative consistency, and visual coherence.

Candidates:
{candidate_summaries}

Return a JSON array of image_ids ordered by relevance."""

    # Call Azure AI Foundry endpoint
    # ... (implementation depends on foundry client setup)
```

## 9. Complete Working Example

This example ties together all components into a cohesive index creation and search workflow:

```python
import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    ComplexField,
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery


def create_candidate_index(index_client: SearchIndexClient, index_name: str):
    """Create the candidate generation search index."""
    fields = [
        SimpleField(name="image_id", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchableField(name="generation_prompt", type=SearchFieldDataType.String),
        SimpleField(name="scene_type", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="lighting_condition", type=SearchFieldDataType.String, filterable=True),
        SearchableField(
            name="tags",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            filterable=True,
            facetable=True,
        ),
        SimpleField(name="emotional_polarity", type=SearchFieldDataType.Double, filterable=True, sortable=True),
        SimpleField(name="low_light_score", type=SearchFieldDataType.Double, filterable=True),
        SearchField(
            name="semantic_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=3072,
            vector_search_profile_name="hnsw-cosine-profile",
        ),
        SearchField(
            name="structural_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1024,
            vector_search_profile_name="hnsw-cosine-profile",
        ),
        SearchField(
            name="style_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=512,
            vector_search_profile_name="hnsw-cosine-profile",
        ),
        SearchField(
            name="primary_character_semantic_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=3072,
            vector_search_profile_name="hnsw-cosine-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw-cosine",
                parameters=HnswParameters(
                    m=4,
                    ef_construction=400,
                    ef_search=500,
                    metric=VectorSearchAlgorithmMetric.COSINE,
                ),
            ),
        ],
        profiles=[
            VectorSearchProfile(
                name="hnsw-cosine-profile",
                algorithm_configuration_name="hnsw-cosine",
            ),
        ],
    )

    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name="semantic-config",
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name="generation_prompt")],
                ),
            ),
        ],
    )

    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )

    return index_client.create_or_update_index(index)


def hybrid_search(
    search_client: SearchClient,
    query_text: str,
    semantic_vec: list[float],
    structural_vec: list[float],
    style_vec: list[float],
    filters: str | None = None,
    top: int = 20,
):
    """Execute a hybrid multi-vector search query."""
    return search_client.search(
        search_text=query_text,
        vector_queries=[
            VectorizedQuery(vector=semantic_vec, k_nearest_neighbors=50, fields="semantic_vector", weight=0.5),
            VectorizedQuery(vector=structural_vec, k_nearest_neighbors=50, fields="structural_vector", weight=0.2),
            VectorizedQuery(vector=style_vec, k_nearest_neighbors=50, fields="style_vector", weight=0.2),
        ],
        filter=filters,
        query_type="semantic",
        semantic_configuration_name="semantic-config",
        facets=["scene_type,count:10", "lighting_condition,count:10", "tags,count:20"],
        top=top,
    )
```

## 10. SDK Version Compatibility Notes

| Feature                  | Minimum SDK Version | Notes                                       |
|--------------------------|---------------------|---------------------------------------------|
| Vector search            | 11.4.0              | Initial vector search support               |
| Multiple vector queries  | 11.4.0              | `vector_queries` parameter on `search()`    |
| VectorizedQuery weight   | 11.5.0              | `weight` parameter for RRF fusion tuning    |
| Semantic ranker v2       | 11.4.0              | `SemanticSearch` configuration model        |
| Pre/post filter modes    | 11.5.0              | `VectorFilterMode` enum                     |
| Scalar quantization      | 11.6.0              | Compressed vector storage                   |
| Narrow data types        | 11.6.0              | `Collection(Edm.Half)`, `Collection(Edm.Int16)` |

> [!IMPORTANT]
> Install version 11.6.0 or later for full feature coverage including compressed vector storage, which is critical for the 10M+ document scale requirement.

```bash
uv add "azure-search-documents>=11.6.0"
```

## 11. Key Findings Summary

1. The `azure-search-documents` SDK provides comprehensive support for multi-vector hybrid search with HNSW indexing and cosine similarity.

2. Multiple vector fields with different dimensionalities (3072, 1024, 512) are supported on a single index by referencing the same or different vector search profiles.

3. Nested complex types do not support vector search. Character vectors must be flattened to top-level fields or stored in a separate index.

4. Hybrid search combines BM25 text search with multiple vector queries using Reciprocal Rank Fusion (RRF). The `weight` parameter on `VectorizedQuery` controls per-field contribution.

5. Scoring profiles apply to text search boosting. Vector weighting is handled separately through `VectorizedQuery.weight`.

6. Batch indexing supports up to 1,000 documents per batch (16 MB limit). For 10M+ documents with high-dimensional vectors, use batch sizes of ~500. Parallel upload with retry logic is essential.

7. Filters execute as pre-filters before vector search by default, meaning `k_nearest_neighbors` returns results from the filtered subset only.

8. Built-in semantic re-ranking is available. Custom re-ranking (emotional alignment, narrative consistency) requires a two-stage retrieval pattern with application-level logic.

9. For compressed storage at scale, SDK 11.6.0+ supports scalar quantization and narrow data types (`Collection(Edm.Half)`), reducing memory footprint by ~50%.

## 12. Recommended Next Research Topics

* Azure AI Foundry Python SDK (`azure-ai-inference`) for embedding generation with `text-embedding-3-large`
* Scalar quantization and narrow vector types for memory optimization at 10M+ scale
* Index alias management for zero-downtime schema migrations
* Azure AI Search pricing tiers and partition/replica configurations for 10M documents with 4 vector fields
* Monitoring and observability integration (Azure Monitor metrics for search service)
* Incremental indexing strategies: change tracking with Azure Blob Storage or Cosmos DB indexers
* `VectorQuery` with `kind="text"` for integrated vectorization (model invocation handled by the search service)
