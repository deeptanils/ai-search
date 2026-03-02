---
title: Image Indexing Process
description: End-to-end walkthrough of how an image is processed, analyzed, embedded, and stored in Azure AI Search
author: AI Search Team
ms.date: 2026-02-27
ms.topic: concept
keywords:
  - image indexing
  - embedding pipeline
  - GPT-4o extraction
  - Azure AI Search
  - Cohere Embed v4
estimated_reading_time: 10
---

## Overview

Each image that enters the pipeline goes through four sequential stages before it becomes a searchable document in Azure AI Search. The stages are input loading, GPT-4o extraction, multi-vector embedding generation, and document upload. A single image produces one index document containing 16 primitive fields and 4 vector fields (20 total).

```text
Image + Prompt
      │
      ▼
┌──────────────────────────────────────────────────┐
│  Stage 1: Input Loading                          │
│  Validate image source (URL or local file)       │
│  Produce an ImageInput object                    │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│  Stage 2: GPT-4o Vision Extraction               │
│  Analyze image visually                          │
│  Return structured ImageExtraction               │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│  Stage 3: Multi-Vector Embedding (4 vectors)     │
│  3 text embeddings + 1 image embedding           │
│  All 4 run in parallel via asyncio.gather        │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│  Stage 4: Document Assembly and Upload           │
│  Combine into SearchDocument (20 fields)         │
│  Upload to Azure AI Search with retry            │
└──────────────────────────────────────────────────┘
```

## Stage 1: Input Loading

Source file: `src/ai_search/ingestion/loader.py`

The pipeline accepts an image from two sources: a public URL or a local file path. Both are normalized into an `ImageInput` object that carries three pieces of information.

| Field               | Type         | Purpose                                    |
|---------------------|--------------|--------------------------------------------|
| `image_id`          | String       | Unique identifier used as the index key    |
| `generation_prompt` | String       | Text prompt that describes or generated the image |
| `image_url`         | String, None | Public URL when loading from a remote source |
| `image_base64`      | String, None | Base64-encoded bytes when loading from a local file |

When you provide a URL, the pipeline stores it directly. When you provide a local file, the pipeline reads the file bytes and base64-encodes them at load time:

```python
# From a URL
image_input = ImageInput.from_url("sample-001", "A night scene in Tokyo", url)

# From a local file (reads bytes, base64-encodes immediately)
image_input = ImageInput.from_file("sample-001", "A night scene in Tokyo", "./photo.jpg")
```

The `ImageInput` also provides `to_openai_image_content()`, which formats the image for the GPT-4o API. URL-based inputs pass the URL directly. File-based inputs use a `data:image/jpeg;base64,...` data URI.

## Stage 2: GPT-4o Vision Extraction

Source file: `src/ai_search/extraction/extractor.py`

The image and its generation prompt are sent to GPT-4o using the structured output API (`client.beta.chat.completions.parse()`). GPT-4o analyzes the image visually and returns a fully typed `ImageExtraction` object.

### What GPT-4o receives

GPT-4o receives:

* A detailed system prompt instructing it to produce specific analyses (semantic, structural, style, characters, metadata, narrative, emotion, objects, low-light metrics).
* The generation prompt as context.
* The image itself via the standard `image_url` content part format.

### What GPT-4o returns

The `ImageExtraction` model contains everything the pipeline needs:

| Field                    | Type                    | Becomes                              |
|--------------------------|-------------------------|--------------------------------------|
| `semantic_description`   | String (~200 words)     | Input to semantic embedding (3072d)  |
| `structural_description` | String (~150 words)     | Input to structural embedding (1024d) |
| `style_description`      | String (~150 words)     | Input to style embedding (512d)      |
| `characters`             | List[CharacterDescription] | `character_count` field in index  |
| `metadata`               | ImageMetadata           | 8 filterable/facetable primitive fields |
| `narrative`              | NarrativeIntent         | `narrative_type` field in index      |
| `emotion`                | EmotionalTrajectory     | `emotional_polarity` field in index  |
| `objects`                | RequiredObjects         | Stored in `extraction_json`          |
| `low_light`              | LowLightMetrics         | `low_light_score` field in index     |

The three description fields are the critical outputs for embedding. Each one captures a different "dimension" of the image's content:

* `semantic_description` covers scene content, subjects, actions, environment, mood, and thematic elements.
* `structural_description` covers spatial composition, layout, object positioning, foreground/midground/background, and geometric structure.
* `style_description` covers artistic style, color palette, lighting, texture, rendering technique, and visual treatment.

### Extraction configuration

| Parameter     | Value                  | Source        |
|---------------|------------------------|---------------|
| Model         | `gpt-4o`               | config.yaml   |
| Temperature   | 0.2                    | config.yaml   |
| Max tokens    | 4096                   | config.yaml   |
| Image detail  | `high`                 | Hardcoded in `to_openai_image_content()` |
| Response format | `ImageExtraction` (Pydantic) | Structured output |

## Stage 3: Multi-Vector Embedding Generation

Source files: `src/ai_search/embeddings/pipeline.py`, `encoder.py`, `semantic.py`, `structural.py`, `style.py`, `image.py`

Four embedding vectors are generated from the extraction output. Three come from text descriptions, one comes from the raw image pixels. All four run concurrently via `asyncio.gather()`.

### The four vectors

| Vector               | Input Source                    | Model                       | Dimensions | Purpose                          |
|----------------------|---------------------------------|-----------------------------|------------|-----------------------------------|
| `semantic_vector`    | `extraction.semantic_description` (text) | text-embedding-3-large  | 3072       | Scene meaning and content similarity |
| `structural_vector`  | `extraction.structural_description` (text) | text-embedding-3-large | 1024       | Composition and layout similarity |
| `style_vector`       | `extraction.style_description` (text)    | text-embedding-3-large  | 512        | Artistic style similarity        |
| `image_vector`       | Raw image pixels (visual)       | Cohere Embed v4 (`embed-v-4-0`) | 1024  | Pixel-level visual similarity    |

### Text embeddings (3 vectors)

The three text vectors all use the same underlying model, `text-embedding-3-large`, but at different dimensionalities. This model supports Matryoshka dimensionality reduction, meaning you can request fewer dimensions and the model truncates to that size while preserving the most important signal.

```text
semantic_description  ──► text-embedding-3-large (dims=3072) ──► semantic_vector
structural_description ──► text-embedding-3-large (dims=1024) ──► structural_vector
style_description     ──► text-embedding-3-large (dims=512)  ──► style_vector
```

Each text string passes through `embed_text()` in `encoder.py`, which calls the Azure OpenAI embeddings API via `AsyncAzureOpenAI.embeddings.create()`. Authentication uses Entra ID with `DefaultAzureCredential`.

### Image embedding (1 vector)

The image embedding is fundamentally different from the text embeddings. It processes the raw image pixels through Cohere Embed v4, producing a vector in a shared image-text embedding space.

The process involves several steps:

1. If the image comes from a URL, download it asynchronously via `httpx`.
2. Resize the image to fit within 512x512 pixels using Pillow's `Image.thumbnail()` with Lanczos resampling. This reduces token consumption and avoids S0-tier rate limits.
3. Convert to RGB mode if necessary (handles RGBA, palette, or grayscale inputs).
4. Re-encode as JPEG at 80% quality.
5. Base64-encode the resized bytes into a `data:image/jpeg;base64,...` data URI.
6. Send to Cohere Embed v4 via `ImageEmbeddingsClient.embed()` with `ImageEmbeddingInput`.

```text
Raw image (any size)
      │
      ▼
Resize to max 512×512 (LANCZOS)
      │
      ▼
Convert to RGB + JPEG @ 80%
      │
      ▼
Base64 encode → data URI
      │
      ▼
ImageEmbeddingsClient.embed(input=[ImageEmbeddingInput(image=data_uri)])
      │
      ▼
image_vector (1024 floats)
```

> [!IMPORTANT]
> The image embedding uses `ImageEmbeddingsClient` (not `EmbeddingsClient`). Using the wrong client causes the model to tokenize the base64 string as text instead of processing it visually, producing meaningless vectors.

### Vector validation

Every vector passes through `_validate_vector()` after generation. This function checks that the returned vector is not None, not empty, and matches the expected number of dimensions. A mismatch raises a `ValueError` with diagnostic information.

### Parallel execution

All four embedding calls happen simultaneously:

```python
results = await asyncio.gather(
    generate_semantic_vector(extraction.semantic_description),     # 3072d
    generate_structural_vector(extraction.structural_description), # 1024d
    generate_style_vector(extraction.style_description),           # 512d
    embed_image(image_url=image_url, image_bytes=image_bytes),     # 1024d
)
```

This cuts total embedding time from ~4 sequential API round-trips to ~1 round-trip (limited by the slowest call, typically the image embedding).

## Stage 4: Document Assembly and Upload

Source file: `src/ai_search/indexing/indexer.py`

### Document assembly

`build_search_document()` combines the `ImageInput`, `ImageExtraction`, and `ImageVectors` into a single `SearchDocument` with 20 fields:

#### 16 primitive fields

| Field                | Source                              | Index Type                    |
|----------------------|-------------------------------------|-------------------------------|
| `image_id`           | `image_input.image_id`              | String (key, filterable)      |
| `generation_prompt`  | `image_input.generation_prompt`     | String (searchable)           |
| `image_url`          | `image_input.image_url`             | String                        |
| `scene_type`         | `extraction.metadata.scene_type`    | String (filterable, facetable) |
| `time_of_day`        | `extraction.metadata.time_of_day`   | String (filterable)           |
| `lighting_condition` | `extraction.metadata.lighting_condition` | String (filterable, facetable) |
| `primary_subject`    | `extraction.metadata.primary_subject` | String (filterable)         |
| `artistic_style`     | `extraction.metadata.artistic_style` | String (filterable, facetable) |
| `tags`               | `extraction.metadata.tags`          | Collection(String) (searchable, filterable, facetable) |
| `narrative_theme`    | `extraction.metadata.narrative_theme` | String (filterable)         |
| `narrative_type`     | `extraction.narrative.narrative_type` | String (filterable)         |
| `emotional_polarity` | `extraction.emotion.emotional_polarity` | Double (filterable, sortable) |
| `low_light_score`    | `extraction.low_light.brightness_score` | Double (filterable)       |
| `character_count`    | `len(extraction.characters)`        | Int32 (filterable, sortable)  |
| `metadata_json`      | Full metadata object serialized as JSON | String                    |
| `extraction_json`    | Full extraction object serialized as JSON | String                  |

The `metadata_json` and `extraction_json` fields store the complete GPT-4o output as JSON strings. This preserves all extracted data (characters, objects, narrative details, low-light sub-scores) that would otherwise be lost, since only selected fields are promoted to top-level index fields.

#### 4 vector fields

| Field               | Dimensions | Retrievable | HNSW Algorithm |
|---------------------|------------|-------------|----------------|
| `semantic_vector`   | 3072       | No (hidden) | cosine, m=4, ef_construction=400, ef_search=500 |
| `structural_vector` | 1024       | No (hidden) | cosine, m=4, ef_construction=400, ef_search=500 |
| `style_vector`      | 512        | No (hidden) | cosine, m=4, ef_construction=400, ef_search=500 |
| `image_vector`      | 1024       | Yes         | cosine, m=4, ef_construction=400, ef_search=500 |

Only `image_vector` is marked as retrievable. This is required for image-to-image search, which retrieves all stored image vectors and computes exact cosine similarity client-side. The text vectors are only queried via Azure AI Search's built-in HNSW approximate nearest-neighbor search and never need to be retrieved directly.

### Document upload

`upload_documents()` serializes each `SearchDocument` to a dictionary via `model_dump()`, strips empty vector fields (to avoid uploading zero-length arrays), and calls `SearchClient.upload_documents()`.

Upload features:

* Batching in chunks of 500 documents (configurable via `batch.index_batch_size` in config.yaml).
* Exponential backoff retry on HTTP 429 (rate limit) and 503 (service unavailable) errors.
* Up to 3 retry attempts per batch with delays of 1s, 2s, and 4s.
* Per-batch success counting and structured logging.

### What a final indexed document looks like

```text
┌──────────────────────────────────────────────────────────────┐
│ image_id:           "sample-001"                             │
│ generation_prompt:  "A cinematic night scene of a woman..."  │
│ image_url:          "https://images.unsplash.com/..."        │
│ scene_type:         "Urban Night Scene"                      │
│ time_of_day:        "Night"                                  │
│ lighting_condition: "Neon / Artificial"                      │
│ primary_subject:    "Woman in red dress"                     │
│ artistic_style:     "Film Noir"                              │
│ tags:               ["cinematic", "night", "neon", "rain"]   │
│ narrative_theme:    "Urban solitude"                         │
│ narrative_type:     "cinematic"                              │
│ emotional_polarity: 0.3                                      │
│ low_light_score:    0.35                                     │
│ character_count:    1                                        │
│ metadata_json:      "{\"scene_type\": \"Urban Night...\"}"   │
│ extraction_json:    "{\"semantic_description\": \"...\"}"    │
│                                                              │
│ semantic_vector:    [0.012, -0.034, ..., 0.056]    (3072d)   │
│ structural_vector:  [0.045, 0.078, ..., -0.023]    (1024d)   │
│ style_vector:       [-0.011, 0.067, ..., 0.089]    ( 512d)   │
│ image_vector:       [0.033, -0.021, ..., 0.044]    (1024d)   │
└──────────────────────────────────────────────────────────────┘
```

All 20 fields reside in a single document within one Azure AI Search index. Each image produces exactly one document.

## Batch Ingestion

Source file: `scripts/ingest_samples.py`

For processing multiple images, the batch ingestion script adds concurrency controls and rate-limit handling on top of the same four stages:

| Parameter            | Value | Purpose                                       |
|----------------------|-------|-----------------------------------------------|
| `MAX_CONCURRENT`     | 3     | Maximum parallel image pipelines               |
| `INTER_IMAGE_DELAY_S`| 2     | Delay between starting each image (seconds)   |
| `MAX_RETRIES`        | 5     | Retry count on HTTP 429 errors                |
| `RETRY_BACKOFF_S`    | 30    | Base backoff delay between retries (seconds)  |

The script reads image definitions from `data/sample_images.json` and processes each through the full pipeline. Already-indexed images are skipped unless `--force` is passed.

## How the Indexed Data Supports Search

The 20 fields serve two distinct search strategies:

### Text search (SearchMode.TEXT)

Uses the 3 text vectors plus BM25 full-text search, fused through Reciprocal Rank Fusion (RRF):

* `semantic_vector` is searched with the embedded query text at 3072 dimensions.
* `structural_vector` is searched at 1024 dimensions.
* `style_vector` is searched at 512 dimensions.
* BM25 runs against `generation_prompt` and `tags` (with the `text-boost` scoring profile weighting prompts 3x and tags 2x).
* All four result sets merge via RRF, then scores normalize to 0-1 via min-max scaling.

### Image search (SearchMode.IMAGE)

Uses only `image_vector`:

* The query image is embedded via Cohere Embed v4 (same pipeline as Stage 3).
* All stored `image_vector` values are retrieved from the index (possible because `image_vector` is marked `retrievable=True`).
* Exact cosine similarity is computed client-side with NumPy, bypassing HNSW approximation.
* Results are sorted by true cosine similarity scores (0-1).

The primitive fields (`scene_type`, `tags`, `lighting_condition`, etc.) support OData filtering and faceted navigation in both search modes.

## Source File Reference

| File                              | Role                                         |
|-----------------------------------|----------------------------------------------|
| `src/ai_search/ingestion/loader.py`   | Image input loading and validation       |
| `src/ai_search/extraction/extractor.py` | GPT-4o vision extraction              |
| `src/ai_search/ingestion/metadata.py` | Standalone metadata generation (alternative) |
| `src/ai_search/embeddings/pipeline.py` | Parallel embedding orchestrator        |
| `src/ai_search/embeddings/encoder.py` | Base text embedding via text-embedding-3-large |
| `src/ai_search/embeddings/semantic.py` | Semantic vector (3072d)                |
| `src/ai_search/embeddings/structural.py` | Structural vector (1024d)            |
| `src/ai_search/embeddings/style.py`   | Style vector (512d)                      |
| `src/ai_search/embeddings/image.py`   | Image vector via Cohere Embed v4         |
| `src/ai_search/indexing/schema.py`    | Index schema definition (20 fields)      |
| `src/ai_search/indexing/indexer.py`   | Document assembly and batch upload       |
| `src/ai_search/models.py`            | Pydantic data models                     |
| `scripts/ingest_samples.py`          | Batch ingestion with rate-limit handling |
