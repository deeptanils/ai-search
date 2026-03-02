---
title: Search Process
description: End-to-end walkthrough of how text-to-image and image-to-image search queries are processed, vectorized, and ranked via multi-vector RRF reranking
author: AI Search Team
ms.date: 2026-03-02
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

| Vector Target      | Input to Embedding                                                                                                                                                                                                                                                                                                                         |
|--------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Semantic (3072d)   | Raw query text, unchanged                                                                                                                                                                                                                                                                                                                 |
| Structural (1024d) | GPT-4o output from cinematic composition analyst prompt: character count and positioning, scale relationships (giant vs normal, divine vs mortal), camera angle (low-angle heroic, aerial, close-up, wide panorama), foreground/midground/background layering, action direction and dynamic lines, depth staging, focal point, symmetry     |
| Style (512d)       | GPT-4o output from cinematic art director prompt: scene-specific color palette (golden/fiery reds for battles, cool blues for divine visions), lighting type (divine glow, dramatic backlighting, firelight), texture (ornamental jewelry, battle-worn armor), atmosphere (misty, smoky), rendering approach, cultural art conventions       |

Both GPT-4o calls run in parallel via `asyncio.gather()` (each takes ~200 tokens). Temperature is set to 0.2 for deterministic output. The prompts are tailored for Indian mythology and cultural artwork, referencing cinematic shot composition conventions.

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
│    ├── semantic_description   (~150 words)        │
│    ├── structural_description (~80 words)         │
│    └── style_description      (~80 words)         │
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

The raw image bytes are resized to a configurable maximum size (default 768 pixels, from `config.query_extraction.max_image_size`), base64-encoded, and sent to GPT-4o as a structured output request using the `QueryImageDescriptions` response format. The system prompt acts as a cinematic image analyst, instructing GPT-4o to identify characters by proper name, name the story episode, and focus on dimensions that distinguish the scene from other scenes with the same characters.

All query extraction parameters are configurable via `config.yaml` under `query_extraction`:

| Parameter        | Default | Purpose                                     |
|------------------|---------|---------------------------------------------|
| `image_detail`   | high    | GPT-4o vision detail level                  |
| `temperature`    | 0.0     | Deterministic output for cache consistency  |
| `max_tokens`     | 1200    | Maximum response tokens                     |
| `max_image_size` | 768     | Max pixel dimension for resizing            |

GPT-4o returns three descriptions with cinematic dimensions:

| Description              | Content                                                                                                                                                                                                 | Length     |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| `semantic_description`   | Named characters, specific actions (flying, fighting, kneeling), weapons/props (gada, bow, mountain, torch), costumes/jewelry, character form/avatar, location, story episode, emotional tone           | ~150 words |
| `structural_description` | Character count and positioning, scale relationships (giant vs normal, divine vs mortal), camera angle, foreground/background separation, action direction, focal point                                  | ~80 words  |
| `style_description`      | Color palette (golden, fiery reds, cool blues), lighting type (divine glow, dramatic backlight, firelight), texture (ornamental, painterly), atmosphere (misty, smoky, radiant), rendering approach      | ~80 words  |

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

The image embedding pipeline resizes the query image to a configurable maximum size (default 768 pixels from `config.query_extraction.max_image_size` for description extraction; the Cohere embedding path uses its own resize logic), converts to RGB JPEG at 80% quality, base64-encodes it, and sends it to `ImageEmbeddingsClient.embed()`.

### Step 3: Multi-vector search with RRF

The 4 query vectors are passed to the same `execute_vector_search()` function. The only difference from text search is the additional `image_vector` query.

Image search uses a separate weight profile (`config.image_search`) that heavily favors the direct image vector, since pixel-level visual similarity is the strongest signal when the user provides an actual image:

| Vector Field       | Config Weight | Search Weight (×10) |
|--------------------|---------------|---------------------|
| `semantic_vector`  | 0.15          | 1.5                 |
| `structural_vector`| 0.05          | 0.5                 |
| `style_vector`     | 0.05          | 0.5                 |
| `image_vector`     | 0.65          | 6.5                 |

Azure AI Search runs 4 HNSW queries, retrieves candidates from each, and fuses them via RRF with the specified weights. The image vector at 65% weight dominates the ranking, while the three text vectors contribute contextual semantic, compositional, and stylistic similarity derived from the GPT-4o descriptions.

### SHA-256 result cache

Image search results are cached by a SHA-256 hash of the raw image bytes. The cache holds up to 64 entries with FIFO eviction. Temperature is set to 0.0 for the GPT-4o extraction call, which combined with the cache ensures that uploading the same image always produces identical rankings. This eliminates non-determinism from GPT-4o description generation.

### Step 4: Score normalization

Identical to text search: min-max normalization to [0, 1].

## Comparison of Search Modes

| Aspect                | Text Search (TEXT)                                               | Image Search (IMAGE)                                                   |
|-----------------------|------------------------------------------------------------------|------------------------------------------------------------------------|
| User input            | Free-text query string                                           | Raw image bytes (JPEG/PNG)                                             |
| Query expansion       | GPT-4o generates cinematic structural and style descriptions     | GPT-4o extracts cinematic semantic, structural, and style descriptions |
| Vectors generated     | 3 (semantic, structural, style)                                  | 4 (semantic, structural, style, image)                                 |
| Image embedding       | Not used                                                         | Cohere Embed v4 on raw pixels                                          |
| Dominant weight       | semantic at 0.40                                                 | image at 0.65                                                          |
| Search execution      | 3-vector RRF                                                     | 4-vector RRF                                                           |
| Result caching        | Not cached                                                       | SHA-256 keyed, max 64 entries                                          |
| BM25 keyword search   | Not used                                                         | Not used                                                               |
| Score normalization   | Min-max [0, 1]                                                   | Min-max [0, 1]                                                         |
| OData filter support  | Yes                                                              | No                                                                     |

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

Weights are configured in `config.yaml` and control the relative importance of each vector in RRF fusion. Two separate weight profiles exist — one for text search and one for image search — selected automatically based on `SearchMode`:

```yaml
# Text-to-image search (semantic meaning is strongest signal)
search:
  semantic_weight: 0.4
  structural_weight: 0.15
  style_weight: 0.15
  image_weight: 0.2         # Not used in text mode (no image vector)
  keyword_weight: 0.1       # Reserved (not used in current pipeline)

# Image-to-image search (direct pixel similarity dominates)
image_search:
  semantic_weight: 0.15
  structural_weight: 0.05
  style_weight: 0.05
  image_weight: 0.65        # Dominant signal in image mode
  keyword_weight: 0.1       # Reserved (not used in current pipeline)
```

The code selects the profile via `config.image_search if search_mode == SearchMode.IMAGE else config.search`. All weights are multiplied by 10 before passing to Azure AI Search to preserve ratio precision in the `VectorizedQuery.weight` parameter.

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

| Parameter          | Value  | Purpose                                                 |
|--------------------|--------|---------------------------------------------------------|
| `m`                | 4      | Number of bi-directional links per node (graph density) |
| `ef_construction`  | 400    | Search width during index build (recall quality)        |
| `ef_search`        | 500    | Search width during queries (query-time recall)         |
| `metric`           | cosine | Distance metric for similarity comparison               |

These values are optimized for the Basic tier of Azure AI Search, balancing recall quality with memory and latency constraints.

## Text-Boost Scoring Profile

The index defines a `text-boost` scoring profile that amplifies BM25 text-match scores for selected fields. This profile is set as the default and takes effect when `search_text` is provided (currently reserved for the legacy `execute_hybrid_search()` path).

| Field              | Boost Weight | Purpose                                      |
|--------------------|--------------|----------------------------------------------|
| `generation_prompt`| 3.0          | Original image generation prompt             |
| `tags`             | 3.0          | Content tags from extraction                 |
| `character_action` | 2.5          | Primary action verb (flying, fighting, etc.) |
| `weapons_props`    | 2.0          | Weapons and props visible in the scene       |

The scoring profile does not affect pure vector search results. It influences ranking only when BM25 keyword matching is active.

## Index Schema

The index `candidate-index` contains 24 fields: 20 primitive fields and 4 vector fields.

### Primitive fields

| Field                | Type              | Searchable | Filterable | Facetable | Purpose                                      |
|----------------------|-------------------|------------|------------|-----------|----------------------------------------------|
| `image_id`           | String (key)      | No         | Yes        | No        | Unique document identifier                   |
| `generation_prompt`  | String            | Yes        | No         | No        | Original image generation prompt             |
| `scene_type`         | String            | No         | Yes        | Yes       | Scene classification                         |
| `time_of_day`        | String            | No         | Yes        | No        | Time of day in the scene                     |
| `lighting_condition` | String            | No         | Yes        | Yes       | Lighting conditions                          |
| `primary_subject`    | String            | No         | Yes        | No        | Main subject of the image                    |
| `character_action`   | String            | Yes        | Yes        | Yes       | Primary action verb (flying, fighting, etc.) |
| `weapons_props`      | Collection(String)| Yes        | Yes        | Yes       | Weapons and props visible                    |
| `location_name`      | String            | No         | Yes        | Yes       | Named location (Lanka, Ayodhya, etc.)        |
| `episode_name`       | String            | No         | Yes        | Yes       | Named story episode (Lanka Dahan, etc.)      |
| `artistic_style`     | String            | No         | Yes        | Yes       | Artistic style classification                |
| `tags`               | Collection(String)| Yes        | Yes        | Yes       | Content tags from extraction (10-15 keywords)|
| `narrative_theme`    | String            | No         | Yes        | No        | Narrative theme                              |
| `narrative_type`     | String            | No         | Yes        | No        | Narrative type (cinematic, fantasy, etc.)     |
| `emotional_polarity` | Double            | No         | Yes        | No        | Emotional polarity score [-1.0, 1.0]         |
| `low_light_score`    | Double            | No         | Yes        | No        | Low-light visibility confidence              |
| `character_count`    | Int32             | No         | Yes        | No        | Number of characters in the scene            |
| `image_url`          | String            | No         | No         | No        | Blob storage URL                             |
| `metadata_json`      | String            | No         | No         | No        | Full metadata as JSON string                 |
| `extraction_json`    | String            | No         | No         | No        | Full extraction as JSON string               |

### Vector fields

| Field                | Dimensions | Profile            | Retrievable | Purpose                          |
|----------------------|------------|--------------------|-------------|----------------------------------|
| `semantic_vector`    | 3072       | hnsw-cosine-profile| No          | Scene meaning                    |
| `structural_vector`  | 1024       | hnsw-cosine-profile| No          | Composition and layout           |
| `style_vector`       | 512        | hnsw-cosine-profile| No          | Artistic style                   |
| `image_vector`       | 1024       | hnsw-cosine-profile| Yes         | Direct pixel similarity (Cohere) |

## Output Format

Both search modes return a list of `SearchResult` objects:

| Field              | Type         | Description                          |
|--------------------|--------------|--------------------------------------|
| `image_id`         | String       | Unique identifier of the matched image |
| `search_score`     | Float        | Normalized score in [0, 1]           |
| `generation_prompt`| String, None | Original generation prompt           |
| `image_url`        | String, None | URL of the indexed image             |
| `scene_type`       | String, None | Scene classification from extraction |
| `tags`             | List[String] | Content tags from extraction         |

Internally, the search retrieves additional cinema-specific fields (`character_action`, `weapons_props`, `location_name`, `episode_name`, and others) from the index for diagnostics and debugging, but these are not surfaced in the `SearchResult` model returned to callers.

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
| `src/ai_search/config.py`              | Config loading, weight profiles, extraction params|
| `src/ai_search/ui/app.py`              | Gradio web UI with text and image search tabs     |
| `config.yaml`                          | Vector weights, HNSW params, retrieval settings   |
