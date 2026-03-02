---
title: Search Process
description: End-to-end walkthrough of how text-to-image and image-to-image search queries are processed, vectorized, and ranked via multi-vector RRF reranking
author: AI Search Team
ms.date: 2026-02-27
ms.topic: concept
keywords:
  - search pipeline
  - multi-vector search
  - RRF reranking
  - text-to-image
  - image-to-image
  - Azure AI Search
estimated_reading_time: 10
---

## Overview

The search pipeline supports two modes, both built on the same underlying multi-vector search with Reciprocal Rank Fusion (RRF) reranking. The unified entry point is `search()` in `pipeline.py`, which dispatches to the appropriate path based on `SearchMode`.

```text
search(mode, query_text | query_image_bytes)
      │
      ├── SearchMode.TEXT  ──► 3 text vectors ──► RRF reranking ──► results
      │
      └── SearchMode.IMAGE ──► 4 vectors (3 text + 1 image) ──► RRF reranking ──► results
```

Both modes produce a ranked list of `SearchResult` objects with scores normalized to the 0-1 range.

## Text-to-Image Search (SearchMode.TEXT)

A user provides a free-text query. The pipeline converts it into 3 query vectors, runs each against its corresponding index vector field via HNSW approximate nearest neighbor search, and Azure AI Search fuses the ranked lists via RRF.

### Flow diagram

```text
User query: "cinematic night scene with neon lights"
      │
      ▼
┌──────────────────────────────────────────────────┐
│  Step 1: Query Expansion (GPT-4o)                │
│                                                  │
│  Raw query ──────────────────► semantic input     │
│  GPT-4o(structural prompt) ──► structural input   │
│  GPT-4o(style prompt) ───────► style input        │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│  Step 2: Text Embedding (parallel)               │
│                                                  │
│  semantic input   ──► text-embedding-3-large ──► 3072d vector │
│  structural input ──► text-embedding-3-large ──► 1024d vector │
│  style input      ──► text-embedding-3-large ──►  512d vector │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│  Step 3: Multi-Vector Search (Azure AI Search)   │
│                                                  │
│  3× VectorizedQuery (one per vector field)       │
│  search_text = None (pure vector, no BM25)       │
│  Each query retrieves top-k_nearest via HNSW     │
│                                                  │
│  Azure fuses results via RRF reranking           │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│  Step 4: Score Normalization                     │
│                                                  │
│  Min-max normalize to [0, 1]                     │
│  Top result = 1.0, others scaled proportionally  │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
              SearchResult[]
```

### Step 1: Query expansion

Source file: `src/ai_search/retrieval/query.py`

The raw query text feeds directly into the semantic embedding. For structural and style embeddings, the pipeline uses GPT-4o to generate specialized descriptions.

| Vector Target   | Input to Embedding                                              |
|------------------|-----------------------------------------------------------------|
| Semantic (3072d) | Raw query text, unchanged                                      |
| Structural (1024d) | GPT-4o output given the structural prompt: "Describe the spatial composition, layout, and geometric structure implied by this query..." |
| Style (512d)     | GPT-4o output given the style prompt: "Describe the artistic style, color palette, lighting, and visual treatment implied by this query..." |

The two GPT-4o calls run sequentially (each takes ~200 tokens). Temperature is set to 0.2 for deterministic output.

### Step 2: Text embedding

Source file: `src/ai_search/embeddings/encoder.py`

All three descriptions are embedded in parallel via `asyncio.gather()`. Each calls `text-embedding-3-large` at its target dimensionality using Matryoshka truncation:

| Description  | Model                    | Dimensions | Output         |
|--------------|--------------------------|------------|----------------|
| Semantic     | text-embedding-3-large   | 3072       | `semantic_vector` |
| Structural   | text-embedding-3-large   | 1024       | `structural_vector` |
| Style        | text-embedding-3-large   | 512        | `style_vector`  |

### Step 3: Multi-vector search with RRF

Source file: `src/ai_search/retrieval/search.py`

The 3 query vectors are passed to `execute_vector_search()`, which builds one `VectorizedQuery` per vector and calls `SearchClient.search()` with `search_text=None` (pure vector search, no BM25 keyword matching).

Each `VectorizedQuery` specifies:

* `vector`: The query embedding.
* `k_nearest_neighbors`: Number of approximate nearest neighbors to retrieve per vector field (default: 100, from `config.retrieval.k_nearest`).
* `fields`: Which index field to search against (`semantic_vector`, `structural_vector`, or `style_vector`).
* `weight`: Relative importance of this vector in the RRF fusion, derived from config weights multiplied by 10.

Configured weights from `config.yaml`:

| Vector Field       | Config Weight | Search Weight (×10) |
|--------------------|---------------|---------------------|
| `semantic_vector`  | 0.40          | 4.0                 |
| `structural_vector`| 0.15          | 1.5                 |
| `style_vector`     | 0.15          | 1.5                 |

Azure AI Search internally runs each vector query against its HNSW index, retrieves the top `k_nearest` candidates, and fuses the three ranked lists using Reciprocal Rank Fusion. RRF produces a single combined score for each document based on its rank position across all vector queries, weighted by the specified weights.

### Step 4: Score normalization

Scores from RRF are relative values that vary in scale. The pipeline applies min-max normalization in-place so the highest-scored document gets 1.0 and the lowest gets 0.0. When all scores are identical, every document receives 1.0.

## Image-to-Image Search (SearchMode.IMAGE)

A user uploads an image. The pipeline extracts 3 text descriptions via GPT-4o vision, embeds each into its text vector space, embeds the raw image into the image vector space, and runs all 4 vectors through the same multi-vector RRF search.

### Flow diagram

```text
User uploads: photo.jpg (raw bytes)
      │
      ▼
┌──────────────────────────────────────────────────┐
│  Step 1: GPT-4o Vision Extraction                │
│                                                  │
│  Image bytes ──► GPT-4o structured output        │
│                                                  │
│  Returns QueryImageDescriptions:                 │
│    ├── semantic_description   (~200 words)        │
│    ├── structural_description (~150 words)        │
│    └── style_description      (~150 words)        │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│  Step 2: 4-Vector Embedding (parallel)           │
│                                                  │
│  semantic_description   ──► text-embedding-3-large ──► 3072d │
│  structural_description ──► text-embedding-3-large ──► 1024d │
│  style_description      ──► text-embedding-3-large ──►  512d │
│  raw image bytes        ──► Cohere Embed v4      ──► 1024d │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│  Step 3: Multi-Vector Search (Azure AI Search)   │
│                                                  │
│  4× VectorizedQuery (one per vector field)       │
│  search_text = None (pure vector, no BM25)       │
│  Each query retrieves top-k_nearest via HNSW     │
│                                                  │
│  Azure fuses results via RRF reranking           │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│  Step 4: Score Normalization                     │
│                                                  │
│  Min-max normalize to [0, 1]                     │
│  Top result = 1.0, others scaled proportionally  │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
              SearchResult[]
```

### Step 1: GPT-4o vision extraction

Source file: `src/ai_search/retrieval/query.py`

The raw image bytes are base64-encoded and sent to GPT-4o as a structured output request using the `QueryImageDescriptions` response format. GPT-4o analyzes the image visually and returns three descriptions:

| Description              | Content                                                          | Length     |
|--------------------------|------------------------------------------------------------------|------------|
| `semantic_description`   | Scene content, subjects, actions, environment, mood, themes      | ~200 words |
| `structural_description` | Spatial composition, layout, object positioning, geometry        | ~150 words |
| `style_description`      | Artistic style, color palette, lighting, texture, visual treatment | ~150 words |

These descriptions mirror the three text descriptions generated during indexing (from `ImageExtraction`), ensuring that query vectors and indexed vectors occupy the same semantic spaces.

### Step 2: 4-vector embedding

Source files: `src/ai_search/embeddings/encoder.py`, `src/ai_search/embeddings/image.py`

All four embeddings run in parallel via `asyncio.gather()`:

| Input                     | Model                    | Dimensions | Output             |
|---------------------------|--------------------------|------------|--------------------|
| `semantic_description`    | text-embedding-3-large   | 3072       | `semantic_vector`  |
| `structural_description`  | text-embedding-3-large   | 1024       | `structural_vector`|
| `style_description`       | text-embedding-3-large   | 512        | `style_vector`     |
| Raw image bytes           | Cohere Embed v4 (`embed-v-4-0`) | 1024 | `image_vector` |

The image embedding pipeline resizes the query image to max 512x512 pixels, converts to RGB JPEG at 80% quality, base64-encodes it, and sends it to `ImageEmbeddingsClient.embed()`.

### Step 3: Multi-vector search with RRF

The 4 query vectors are passed to the same `execute_vector_search()` function. The only difference from text search is the additional `image_vector` query.

Configured weights for image search:

| Vector Field       | Config Weight | Search Weight (×10) |
|--------------------|---------------|---------------------|
| `semantic_vector`  | 0.40          | 4.0                 |
| `structural_vector`| 0.15          | 1.5                 |
| `style_vector`     | 0.15          | 1.5                 |
| `image_vector`     | 0.20          | 2.0                 |

Azure AI Search runs 4 HNSW queries, retrieves candidates from each, and fuses them via RRF with the specified weights. The image vector contributes visual similarity while the three text vectors contribute semantic, compositional, and stylistic similarity derived from the GPT-4o descriptions.

### Step 4: Score normalization

Identical to text search: min-max normalization to [0, 1].

## Comparison of Search Modes

| Aspect                | Text Search (TEXT)              | Image Search (IMAGE)               |
|-----------------------|---------------------------------|--------------------------------------|
| User input            | Free-text query string          | Raw image bytes (JPEG/PNG)           |
| Query expansion       | GPT-4o generates structural and style descriptions from text | GPT-4o extracts semantic, structural, and style descriptions from image |
| Vectors generated     | 3 (semantic, structural, style) | 4 (semantic, structural, style, image) |
| Image embedding       | Not used                        | Cohere Embed v4 on raw pixels        |
| Search execution      | 3-vector RRF                    | 4-vector RRF                         |
| BM25 keyword search   | Not used                        | Not used                             |
| Score normalization   | Min-max [0, 1]                  | Min-max [0, 1]                       |
| OData filter support  | Yes                             | No                                   |

## Reciprocal Rank Fusion (RRF)

RRF is a rank-based fusion method built into Azure AI Search. For each document appearing in any of the individual result lists, RRF computes a combined score based on the document's rank position across all queries.

The formula for each document is:

$$\text{RRF}(d) = \sum_{q \in Q} \frac{w_q}{k + \text{rank}_q(d)}$$

Where:

* $Q$ is the set of vector queries.
* $w_q$ is the weight assigned to query $q$.
* $k$ is a constant (typically 60) that dampens the effect of high-ranked differences.
* $\text{rank}_q(d)$ is the rank of document $d$ in the result list for query $q$.

Documents that rank highly across multiple vector queries receive higher combined scores, even if no single query ranks them at the top. This makes RRF robust to outliers in any individual vector space.

## Vector Weight Configuration

Weights are configured in `config.yaml` under the `search` section and control the relative importance of each vector in RRF fusion:

```yaml
search:
  semantic_weight: 0.4      # Scene meaning (strongest signal)
  structural_weight: 0.15   # Composition and layout
  style_weight: 0.15        # Artistic style
  image_weight: 0.2         # Visual pixel similarity (image mode only)
  keyword_weight: 0.1       # Reserved (not used in current pipeline)
```

These weights are multiplied by 10 before passing to Azure AI Search to preserve ratio precision in the `VectorizedQuery.weight` parameter.

## Relevance Scoring

Source file: `src/ai_search/retrieval/relevance.py`

An optional relevance assessment layer evaluates the statistical distribution of scores across a result set. It assigns a confidence tier based on how much the top result stands out from the rest:

| Tier   | Meaning                                               | Criteria                                      |
|--------|-------------------------------------------------------|-----------------------------------------------|
| HIGH   | Top result is a statistical outlier, very likely match | z-score >= 2.0, gap ratio >= 1%, spread >= 2% |
| MEDIUM | Top result stands out moderately, probable match      | z-score >= 1.3, gap ratio >= 0.5%, spread >= 1.5% |
| LOW    | No result stands out, no confident match              | Below MEDIUM thresholds                       |

Metrics computed:

* z-score: How many standard deviations the top score sits above the mean.
* gap ratio: The difference between #1 and #2 divided by #1.
* spread: The range between the highest and lowest scores.

Relevance filtering is available through the legacy `execute_hybrid_search()` function and discards entire result sets that fall below a specified minimum confidence tier.

## HNSW Configuration

Each vector field in the index uses the same HNSW (Hierarchical Navigable Small World) algorithm for approximate nearest neighbor search:

| Parameter          | Value | Purpose                                          |
|--------------------|-------|--------------------------------------------------|
| `m`                | 4     | Number of bi-directional links per node (graph density) |
| `ef_construction`  | 400   | Search width during index build (recall quality) |
| `ef_search`        | 500   | Search width during queries (query-time recall)  |
| `metric`           | cosine | Distance metric for similarity comparison       |

These values are optimized for the Basic tier of Azure AI Search, balancing recall quality with memory and latency constraints.

## Output Format

Both search modes return a list of `SearchResult` objects:

| Field              | Type        | Description                          |
|--------------------|-------------|--------------------------------------|
| `image_id`         | String      | Unique identifier of the matched image |
| `search_score`     | Float       | Normalized score in [0, 1]           |
| `generation_prompt`| String, None | Original generation prompt           |
| `image_url`        | String, None | URL of the indexed image             |
| `scene_type`       | String, None | Scene classification from extraction |
| `tags`             | List[String] | Content tags from extraction         |

## Source File Reference

| File                                    | Role                                              |
|-----------------------------------------|---------------------------------------------------|
| `src/ai_search/retrieval/pipeline.py`   | Unified `search()` entry point, mode dispatch     |
| `src/ai_search/retrieval/query.py`      | Query vector generation for both modes            |
| `src/ai_search/retrieval/search.py`     | `execute_vector_search()` with RRF reranking      |
| `src/ai_search/retrieval/relevance.py`  | Statistical relevance scoring and filtering       |
| `src/ai_search/embeddings/encoder.py`   | Text embedding via text-embedding-3-large         |
| `src/ai_search/embeddings/image.py`     | Image embedding via Cohere Embed v4               |
| `src/ai_search/models.py`              | `SearchMode` enum and `SearchResult` model        |
| `src/ai_search/ui/app.py`              | Gradio web UI with text and image search tabs     |
| `config.yaml`                          | Vector weights, HNSW params, retrieval settings   |
