# AI Search Pipeline — Architecture

## Overview

This document describes the current architecture optimized for Azure AI Search **Basic tier** (~$75/month), including granular API details, SDK classes, authentication methods, configuration, and the full data flow from image ingestion through hybrid retrieval. The future advanced design for Standard tier follows at the end.

---

## Azure Services and SDK Dependencies

### Services Used

| Service                        | Purpose                                         | SDK / Client                                     |
|--------------------------------|-------------------------------------------------|--------------------------------------------------|
| Azure AI Foundry (OpenAI)      | GPT-4o extraction, text-embedding-3-large       | `openai.AzureOpenAI`, `openai.AsyncAzureOpenAI`  |
| Azure AI Foundry (Inference)   | Cohere Embed v4 text + image embeddings          | `azure.ai.inference.aio.EmbeddingsClient`, `azure.ai.inference.aio.ImageEmbeddingsClient` |
| Azure AI Search                | Index management, document upload, hybrid search | `azure.search.documents.SearchClient`, `azure.search.documents.indexes.SearchIndexClient` |
| Azure Computer Vision 4.0     | Florence image/text vectorization (alt backend)  | `httpx.AsyncClient` (REST API)                   |

### SDK Packages and Versions

From `pyproject.toml`:

| Package                    | Version          | Purpose                                       |
|----------------------------|------------------|-----------------------------------------------|
| `openai`                   | `>=1.58.0`       | Azure OpenAI: GPT-4o + text-embedding-3-large |
| `azure-search-documents`   | `>=11.6.0`       | Azure AI Search index + document operations   |
| `azure-identity`           | `>=1.17.0`       | `DefaultAzureCredential`, bearer token provider |
| `azure-ai-inference`       | `>=1.0.0b9`      | Foundry Inference: `EmbeddingsClient`, `ImageEmbeddingsClient` |
| `aiohttp`                  | `>=3.9`          | Required by `azure-ai-inference` async clients |
| `pydantic`                 | `>=2.0`          | Data models and structured output schemas     |
| `pydantic-settings`        | `>=2.0`          | Environment variable loading for secrets      |
| `pyyaml`                   | `>=6.0`          | `config.yaml` parsing                         |
| `python-dotenv`            | `>=1.0`          | `.env` file loading                           |
| `pillow`                   | `>=10.0`         | Image resizing before embedding               |
| `httpx`                    | `>=0.27`         | Azure Computer Vision REST, image downloads   |
| `structlog`                | `>=24.0`         | Structured logging                            |
| `numpy`                    | `>=1.26`         | Numeric operations for relevance scoring      |

### Authentication Methods

| Service                     | Auth Method                        | Details                                                                              |
|-----------------------------|------------------------------------|--------------------------------------------------------------------------------------|
| Azure OpenAI (Foundry)      | Entra ID (`DefaultAzureCredential`) | `get_bearer_token_provider()` with scope `https://cognitiveservices.azure.com/.default` |
| Azure AI Inference (Foundry) | Entra ID (`DefaultAzureCredential`) | `credential_scopes=["https://cognitiveservices.azure.com/.default"]`                   |
| Azure AI Search             | API Key (`AzureKeyCredential`)     | Admin key from `AZURE_AI_SEARCH_API_KEY` env var                                     |
| Azure Computer Vision       | API Key (HTTP header)              | `Ocp-Apim-Subscription-Key` header from `AZURE_CV_API_KEY`                           |

All client factories are cached with `@lru_cache(maxsize=1)` in `src/ai_search/clients.py` for singleton behavior.

---

## API Calls Reference

### Azure OpenAI API Calls

| SDK Method                              | Model    | Route            | Location                                | Purpose                                 |
|-----------------------------------------|----------|------------------|-----------------------------------------|-----------------------------------------|
| `client.beta.chat.completions.parse()`  | `gpt-4o` | `POST /chat/completions` | `src/ai_search/extraction/extractor.py` | Unified image extraction with structured output (`response_format=ImageExtraction`) |
| `client.beta.chat.completions.parse()`  | `gpt-4o` | `POST /chat/completions` | `src/ai_search/ingestion/metadata.py`   | Synthetic metadata generation (`response_format=ImageMetadata`)                     |
| `client.chat.completions.create()`      | `gpt-4o` | `POST /chat/completions` | `src/ai_search/retrieval/query.py`      | LLM expansion: structural + style descriptions for query vectors                    |
| `client.embeddings.create()`            | `text-embedding-3-large` | `POST /embeddings` | `src/ai_search/embeddings/encoder.py` | Text embedding at configurable Matryoshka dimensions (3072/1024/512)              |

**OpenAI SDK version**: The `openai` Python SDK wraps the Azure OpenAI REST API. API version is configurable via `AZURE_OPENAI_API_VERSION` (default: `2024-12-01-preview`).

### Azure AI Inference API Calls

| SDK Method                           | Model         | Route                    | Location                              | Purpose                               |
|--------------------------------------|---------------|--------------------------|---------------------------------------|---------------------------------------|
| `ImageEmbeddingsClient.embed()`      | `embed-v-4-0` | `POST /images/embeddings` | `src/ai_search/embeddings/image.py`   | Image embedding via `ImageEmbeddingInput(image=data_uri)` |
| `EmbeddingsClient.embed()`           | `embed-v-4-0` | `POST /embeddings`        | `src/ai_search/embeddings/image.py`   | Cross-modal text-to-image embedding   |

**SDK classes imported** in `src/ai_search/clients.py`:

* `azure.ai.inference.aio.EmbeddingsClient` — text route for cross-modal queries
* `azure.ai.inference.aio.ImageEmbeddingsClient` — image route for pixel embeddings
* `azure.ai.inference.models.ImageEmbeddingInput` — wraps base64 data URI for image input

### Azure Computer Vision 4.0 REST Calls (Florence Backend)

| Route                                              | Method | Body                                                    | Purpose                        |
|----------------------------------------------------|--------|---------------------------------------------------------|--------------------------------|
| `/computervision/retrieval:vectorizeImage`          | `POST` | `{"url": image_url}` or binary with `Content-Type: application/octet-stream` | Image-to-vector (1024d)        |
| `/computervision/retrieval:vectorizeText`           | `POST` | `{"text": query_text}`                                  | Text-to-vector in image space (1024d) |

Query parameters: `api-version` (default `2024-02-01`), `model-version` (default `2023-04-15`).

### Azure AI Search SDK Calls

| SDK Method                                     | Location                            | Purpose                                |
|------------------------------------------------|-------------------------------------|----------------------------------------|
| `SearchIndexClient.create_or_update_index()`   | `src/ai_search/indexing/schema.py`  | Create or update the search index      |
| `SearchClient.upload_documents()`              | `src/ai_search/indexing/indexer.py` | Batch document upload with retry       |
| `SearchClient.search()`                        | `src/ai_search/retrieval/search.py` | Hybrid search (BM25 + multi-vector RRF) |
| `SearchClient.get_document()`                  | `scripts/ingest_samples.py`         | Check if document is already indexed   |

---

## Current Architecture (Basic Tier)

### Tier Constraints

| Constraint      | Basic Tier Limit | Current Usage |
|-----------------|------------------|---------------|
| Vector fields   | 5                | 4             |
| Semantic ranker | Not available    | Not used      |
| Storage         | 2 GB             | Within limits |
| Replicas        | 3                | 1             |
| Partitions      | 1                | 1             |

### Index Schema

The index `candidate-index` contains **19 fields**: 15 primitive fields and 4 vector fields.

#### Primitive Fields

| Field                | SDK Class         | Type                | Key | Filterable | Sortable | Facetable | Searchable |
|----------------------|-------------------|---------------------|-----|------------|----------|-----------|------------|
| `image_id`           | `SimpleField`     | `String`            | Yes | Yes        | —        | —         | —          |
| `generation_prompt`  | `SearchableField` | `String`            | —   | —          | —        | —         | Yes        |
| `scene_type`         | `SimpleField`     | `String`            | —   | Yes        | —        | Yes       | —          |
| `time_of_day`        | `SimpleField`     | `String`            | —   | Yes        | —        | —         | —          |
| `lighting_condition` | `SimpleField`     | `String`            | —   | Yes        | —        | Yes       | —          |
| `primary_subject`    | `SimpleField`     | `String`            | —   | Yes        | —        | —         | —          |
| `artistic_style`     | `SimpleField`     | `String`            | —   | Yes        | —        | Yes       | —          |
| `tags`               | `SearchField`     | `Collection(String)` | — | Yes        | —        | Yes       | Yes        |
| `narrative_theme`    | `SimpleField`     | `String`            | —   | Yes        | —        | —         | —          |
| `narrative_type`     | `SimpleField`     | `String`            | —   | Yes        | —        | —         | —          |
| `emotional_polarity` | `SimpleField`     | `Double`            | —   | Yes        | Yes      | —         | —          |
| `low_light_score`    | `SimpleField`     | `Double`            | —   | Yes        | —        | —         | —          |
| `character_count`    | `SimpleField`     | `Int32`             | —   | Yes        | Yes      | —         | —          |
| `metadata_json`      | `SimpleField`     | `String`            | —   | —          | —        | —         | —          |
| `extraction_json`    | `SimpleField`     | `String`            | —   | —          | —        | —         | —          |

> [!NOTE]
> The `tags` field uses `SearchField` (not `SearchableField`) to correctly handle the `Collection(String)` type. `SearchableField` silently ignores the type parameter for collection fields.

#### Vector Fields

| Field                | Dimensions | HNSW Profile         | Source Model          | Source Data                        |
|----------------------|------------|----------------------|-----------------------|------------------------------------|
| `semantic_vector`    | 3072       | `hnsw-cosine-profile` | text-embedding-3-large | `extraction.semantic_description`  |
| `structural_vector`  | 1024       | `hnsw-cosine-profile` | text-embedding-3-large | `extraction.structural_description` |
| `style_vector`       | 512        | `hnsw-cosine-profile` | text-embedding-3-large | `extraction.style_description`     |
| `image_vector`       | 1024       | `hnsw-cosine-profile` | embed-v-4-0 (Cohere Embed v4) | Raw image pixels (base64 data URI) |

#### HNSW Algorithm Configuration

```python
HnswAlgorithmConfiguration(
    name="hnsw-cosine",
    parameters=HnswParameters(
        m=4,                    # Bi-directional links per node
        ef_construction=400,    # Queue size during index build
        ef_search=500,          # Queue size during search
        metric="cosine",        # Distance metric
    ),
)
```

Profile: `hnsw-cosine-profile` — applied to all 4 vector fields.

#### Scoring Profile

```python
ScoringProfile(
    name="text-boost",
    text_weights=TextWeights(
        weights={
            "generation_prompt": 3.0,
            "tags": 2.0,
        }
    ),
)
```

Set as `default_scoring_profile="text-boost"` on the index. Boosts BM25 relevance for `generation_prompt` (3×) and `tags` (2×).

---

## GPT-4o Extraction Pipeline

### Extraction API Call

```python
client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[system_prompt, user_content_with_image],
    response_format=ImageExtraction,   # Pydantic model as structured output
    temperature=0.2,
    max_tokens=4096,
)
```

The user message includes the image as a vision content block:

```python
{"type": "image_url", "image_url": {"url": data_uri_or_url, "detail": "high"}}
```

### Extraction Output Model (`ImageExtraction`)

The GPT-4o system prompt instructs the model to produce:

| Field                     | Type                   | Description                                |
|---------------------------|------------------------|--------------------------------------------|
| `semantic_description`    | `str`                  | 200-word rich scene description            |
| `structural_description`  | `str`                  | 150-word spatial/composition analysis      |
| `style_description`       | `str`                  | 150-word artistic style analysis           |
| `characters`              | `list[CharacterDescription]` | Per-character semantic, emotion, pose |
| `metadata`                | `ImageMetadata`        | Scene type, lighting, tags, etc.           |
| `narrative`               | `NarrativeIntent`      | Story summary, narrative type, tone        |
| `emotion`                 | `EmotionalTrajectory`  | Starting/mid/end emotions, polarity [-1,1] |
| `objects`                 | `RequiredObjects`      | Key, contextual, symbolic objects          |
| `low_light`               | `LowLightMetrics`      | 5 brightness/quality scores (0.0–1.0)      |

### Metadata Generation (Separate Call)

```python
client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[system_prompt, user_content_with_image],
    response_format=ImageMetadata,
    temperature=0.2,
    max_tokens=1000,
)
```

---

## Embedding Pipeline

### Pipeline Flow

```text
ImageExtraction
    ├── semantic_description  ──► text-embedding-3-large (3072d) ──► semantic_vector
    ├── structural_description ──► text-embedding-3-large (1024d) ──► structural_vector
    ├── style_description ──────► text-embedding-3-large (512d)  ──► style_vector
    └── image_bytes ────────────► embed-v-4-0 (1024d)            ──► image_vector
```

### Orchestration

Defined in `src/ai_search/embeddings/pipeline.py`:

1. **Step 1 — Text embeddings** run in **parallel** via `asyncio.gather()`:
   * `generate_semantic_vector()` — 3072 dimensions
   * `generate_structural_vector()` — 1024 dimensions
   * `generate_style_vector()` — 512 dimensions
2. **Step 2 — Image embedding** runs **separately** (rate-limited on S0 Foundry tier):
   * `embed_image()` — 1024 dimensions (only if `image_url` or `image_bytes` provided)

### Text Embeddings (text-embedding-3-large)

| Property        | Value                           |
|-----------------|---------------------------------|
| SDK class       | `openai.AsyncAzureOpenAI`       |
| SDK method      | `client.embeddings.create()`    |
| Model           | `text-embedding-3-large`        |
| Matryoshka dims | 3072, 1024, 512 (configurable)  |
| Batching        | Chunks by `embedding_chunk_size` (default 2048) |
| Input format    | `list[str]` (plain text)        |
| Output format   | `list[list[float]]`             |

Three specialized wrappers in `src/ai_search/embeddings/`:

| Module            | Dimension | Source Text                         |
|-------------------|-----------|-------------------------------------|
| `semantic.py`     | 3072      | `extraction.semantic_description`   |
| `structural.py`   | 1024      | `extraction.structural_description` |
| `style.py`        | 512       | `extraction.style_description`      |

### Image Embeddings

The image embedding backend is configurable via `config.models.image_embedding_model`:

* `"embed-v-4-0"` (default) → Foundry backend (Azure AI Inference SDK)
* `"azure-cv-florence"` → Florence backend (Computer Vision REST)

#### Foundry Backend (Cohere Embed v4)

| Property           | Value                                                              |
|--------------------|--------------------------------------------------------------------|
| Image SDK class    | `azure.ai.inference.aio.ImageEmbeddingsClient`                    |
| Text SDK class     | `azure.ai.inference.aio.EmbeddingsClient`                         |
| Image input        | `ImageEmbeddingInput(image=data_uri)` — base64 `data:image/jpeg;base64,...` |
| Text input         | `list[str]` — for cross-modal text-to-image search                |
| Image method       | `client.embed(input=[ImageEmbeddingInput(...)], model=..., dimensions=...)` |
| Text method        | `client.embed(input=[text], model=..., dimensions=...)`           |
| Dimensions         | 1024 (configurable)                                               |
| Route (image)      | `POST /images/embeddings`                                         |
| Route (text)       | `POST /embeddings`                                                |

#### Florence Backend (Azure Computer Vision 4.0)

| Property       | Value                                                      |
|----------------|------------------------------------------------------------|
| Client         | `httpx.AsyncClient` with `base_url=secrets.endpoint`      |
| Auth header    | `Ocp-Apim-Subscription-Key`                               |
| Image route    | `POST /computervision/retrieval:vectorizeImage`            |
| Text route     | `POST /computervision/retrieval:vectorizeText`             |
| Dimensions     | 1024 (fixed, not configurable)                             |
| API version    | `2024-02-01`                                               |
| Model version  | `2023-04-15`                                               |

### Image Preprocessing

Before embedding, images are resized to reduce payload size and stay within S0-tier rate limits:

| Property      | Value              |
|---------------|--------------------|
| Max size      | 512 × 512 pixels   |
| Algorithm     | `Image.LANCZOS` (Pillow) |
| Color mode    | Converted to `RGB`  |
| Output format | JPEG, quality 80   |
| Constant      | `_MAX_IMAGE_SIZE = 512` |

### Vector Validation

Each generated vector is validated before storage:

* Checks vector is not `None` or empty
* Checks vector length matches expected dimensions
* Raises `ValueError` with diagnostic info on mismatch

---

## Ingestion Pipeline

### Single Image Processing Flow

Defined in `src/ai_search/ingestion/cli.py`:

```text
ImageInput (url or file path)
    │
    ▼
GPT-4o Extraction ──► ImageExtraction (descriptions, metadata, characters)
    │
    ▼
Embedding Pipeline ──► ImageVectors (4 vectors generated)
    │
    ▼
build_search_document() ──► SearchDocument (19 fields)
    │
    ▼
upload_documents([doc]) ──► Azure AI Search (batch upload with retry)
```

### Image Input Handling

`ImageInput` Pydantic model in `src/ai_search/ingestion/loader.py`:

| Field              | Type            | Description                          |
|--------------------|-----------------|--------------------------------------|
| `image_id`         | `str`           | Unique identifier for the document   |
| `generation_prompt` | `str`          | Text prompt describing the image     |
| `image_url`        | `str` or `None` | URL for remote images                |
| `image_base64`     | `str` or `None` | Base64-encoded image for local files |

Factory methods:

* `ImageInput.from_url(image_id, prompt, url)` — URL-based input
* `ImageInput.from_file(image_id, prompt, path)` — local file, auto base64-encodes

Conversion to OpenAI vision format:

```python
{"type": "image_url", "image_url": {"url": data_uri_or_url, "detail": "high"}}
```

### Batch Ingestion

Defined in `scripts/ingest_samples.py`:

| Parameter            | Value        | Purpose                              |
|----------------------|--------------|--------------------------------------|
| `INTER_IMAGE_DELAY_S` | 10 seconds  | Delay between images (S0 rate limit) |
| `MAX_RETRIES`        | 5            | Retry attempts on HTTP 429 errors    |
| `RETRY_BACKOFF_S`    | 30s × attempt | Linear backoff multiplier           |
| Upload batch size    | 500          | Documents per upload batch (from config) |

**Batch process**:

1. Load sample images from `data/sample_images.json`
2. Pre-download all images synchronously (`httpx.Client`) to avoid async SSL issues
3. Check which docs are already indexed via `get_document()` unless `--force` flag is set
4. For each image: extract → embed → build document → upload
5. Sleep `INTER_IMAGE_DELAY_S` between images
6. On HTTP 429: retry with `RETRY_BACKOFF_S × attempt_number` backoff

### Document Upload with Retry

Defined in `src/ai_search/indexing/indexer.py`:

| Parameter      | Value                        |
|----------------|------------------------------|
| Max retries    | 3                            |
| Base delay     | 1.0 second                   |
| Backoff        | Exponential: `1s × 2^attempt` |
| Retry on       | HTTP 429 (rate limited), 503 (service unavailable) |
| Batch size     | `index_batch_size` from config (default 500) |

Empty vector fields (`list` with `len == 0`) are omitted from the upload payload to avoid index errors.

---

## Retrieval Pipeline

### Pipeline Flow

```text
Query Text (+ optional image URL)
    │
    ├── GPT-4o LLM expansion ──► structural_description, style_description
    │       └── temperature=0.2, max_tokens=200
    │
    ├── text-embedding-3-large ──► semantic_vector (3072d)
    ├── text-embedding-3-large ──► structural_vector (1024d)
    ├── text-embedding-3-large ──► style_vector (512d)
    │
    └── embed-v-4-0 ──► image_vector (1024d)
            ├── If query has image URL: embed_image() — visual embedding
            └── If text only: embed_text_for_image_search() — cross-modal text-to-image
    │
    ▼
Azure AI Search Hybrid Query
    ├── BM25 full-text on generation_prompt (with text-boost scoring profile)
    ├── VectorizedQuery × 4 (weighted, k_nearest=100)
    └── RRF fusion ──► ranked results
    │
    ▼
Relevance Scoring ──► Confidence tier (HIGH / MEDIUM / LOW)
    │
    ▼
SearchResult[] (filtered by min_confidence)
```

### Query Vector Generation

Defined in `src/ai_search/retrieval/query.py`:

1. **LLM expansion** generates structural and style descriptions from the query text using GPT-4o (`temperature=0.2`, `max_tokens=200`)
2. **Parallel text embedding** via `asyncio.gather()` produces semantic (3072d), structural (1024d), and style (512d) vectors
3. **Image-space vector**: if a query image URL is provided, `embed_image()` is used; otherwise `embed_text_for_image_search()` performs cross-modal text-to-image embedding

### Hybrid Search Execution

Defined in `src/ai_search/retrieval/search.py`:

```python
VectorizedQuery(
    vector=query_vectors["semantic_vector"],
    k_nearest_neighbors=100,      # from config.retrieval.k_nearest
    fields="semantic_vector",
    weight=4.0,                   # config weight 0.4 × 10
)
```

**Weight scaling**: config weights are multiplied by 10 to preserve ratio versus BM25's implicit weight of 1.0.

| Vector Field        | Config Weight | Effective Weight (×10) |
|---------------------|---------------|------------------------|
| `semantic_vector`   | 0.4           | 4.0                    |
| `structural_vector` | 0.15          | 1.5                    |
| `style_vector`      | 0.15          | 1.5                    |
| `image_vector`      | 0.2           | 2.0                    |
| BM25 text           | 0.1           | implicit 1.0           |

**Search call**:

```python
client.search(
    search_text=query_text,
    vector_queries=[semantic_vq, structural_vq, style_vq, image_vq],
    filter=odata_filter,           # Optional OData filter expression
    select=SELECT_FIELDS,
    top=50,                        # from config.retrieval.top_k
)
```

**SELECT_FIELDS**: `image_id`, `generation_prompt`, `scene_type`, `tags`, `narrative_type`, `emotional_polarity`, `low_light_score`, `character_count`, `extraction_json`, `metadata_json`

### Relevance Scoring

Defined in `src/ai_search/retrieval/relevance.py`:

Cosine similarity with high-dimensional embeddings produces uniformly high scores (0.95+). Absolute thresholds cannot distinguish true matches from noise in small corpora. The relevance module uses **relative metrics** across the result set:

| Metric      | Formula                     | Purpose                          |
|-------------|-----------------------------|---------------------------------|
| `gap`       | `top1 - top2`               | Absolute score gap               |
| `gap_ratio` | `gap / top1`                | Relative separation from runner-up |
| `z_score`   | `(top1 - mean) / stdev`    | Statistical outlier detection    |
| `spread`    | `max - min`                 | Score range across results       |

**Confidence tiers**:

| Tier       | z_score | gap_ratio | spread | Meaning                          |
|------------|---------|-----------|--------|----------------------------------|
| **HIGH**   | ≥ 2.0   | ≥ 0.01    | ≥ 0.02 | Clear outlier, likely true match |
| **MEDIUM** | ≥ 1.3   | ≥ 0.005   | ≥ 0.015 | Moderately distinct, probable match |
| **LOW**    | below thresholds | — | —    | No confident match               |

Returns `RelevanceResult` dataclass with: `confidence`, `top_score`, `gap`, `gap_ratio`, `z_score`, `spread`, `mean`, `stdev`.

---

## Configuration

### config.yaml

```yaml
models:
  embedding_model: text-embedding-3-large
  llm_model: gpt-4o
  image_embedding_model: embed-v-4-0

search:
  semantic_weight: 0.4
  structural_weight: 0.15
  style_weight: 0.15
  image_weight: 0.2
  keyword_weight: 0.1

index:
  name: candidate-index
  vector_dimensions:
    semantic: 3072
    structural: 1024
    style: 512
    image: 1024
  hnsw:
    m: 4
    ef_construction: 400
    ef_search: 500

retrieval:
  top_k: 50
  k_nearest: 100

extraction:
  image_detail: high
  temperature: 0.2
  max_tokens: 4096

batch:
  index_batch_size: 500
  embedding_chunk_size: 2048
  max_concurrent_requests: 50
```

### Pydantic Config Model Hierarchy

```text
AppConfig (BaseModel)
├── models: ModelsConfig
│   ├── embedding_model: str = "text-embedding-3-large"
│   ├── llm_model: str = "gpt-4o"
│   └── image_embedding_model: str = "embed-v-4-0"
├── search: SearchWeightsConfig
│   ├── semantic_weight: float = 0.4
│   ├── structural_weight: float = 0.15
│   ├── style_weight: float = 0.15
│   ├── image_weight: float = 0.2
│   └── keyword_weight: float = 0.1
├── index: IndexConfig
│   ├── name: str = "candidate-index"
│   ├── vector_dimensions: VectorDimensionsConfig
│   │   ├── semantic: int = 3072
│   │   ├── structural: int = 1024
│   │   ├── style: int = 512
│   │   └── image: int = 1024
│   └── hnsw: HnswConfig
│       ├── m: int = 4
│       ├── ef_construction: int = 400
│       └── ef_search: int = 500
├── retrieval: RetrievalConfig
│   ├── top_k: int = 50
│   └── k_nearest: int = 100
├── extraction: ExtractionConfig
│   ├── image_detail: str = "high"
│   ├── temperature: float = 0.2
│   └── max_tokens: int = 4096
└── batch: BatchConfig
    ├── index_batch_size: int = 500
    ├── embedding_chunk_size: int = 2048
    └── max_concurrent_requests: int = 50
```

### Environment Variables

All loaded via `pydantic-settings` with `SettingsConfigDict(env_file=".env")`:

| Variable                        | Secret Class                 | Required | Default                | Purpose                           |
|---------------------------------|------------------------------|----------|------------------------|-----------------------------------|
| `AZURE_FOUNDRY_ENDPOINT`         | `AzureFoundrySecrets`        | **Yes**  | —                      | Azure AI Foundry base endpoint    |
| `AZURE_FOUNDRY_EMBED_ENDPOINT`   | `AzureFoundrySecrets`        | No       | `None`                 | Foundry models endpoint for Inference SDK |
| `AZURE_FOUNDRY_API_KEY`          | `AzureFoundrySecrets`        | No       | `None`                 | API key (fallback, Entra ID preferred)  |
| `AZURE_OPENAI_API_VERSION`       | `AzureOpenAISecrets`         | No       | `2024-12-01-preview`   | OpenAI API version                |
| `AZURE_AI_SEARCH_ENDPOINT`       | `AzureSearchSecrets`         | **Yes**  | —                      | Azure AI Search endpoint          |
| `AZURE_AI_SEARCH_API_KEY`        | `AzureSearchSecrets`         | **Yes**  | —                      | Azure AI Search admin API key     |
| `AZURE_AI_SEARCH_INDEX_NAME`     | `AzureSearchSecrets`         | No       | `candidate-index`      | Index name override               |
| `AZURE_CV_ENDPOINT`              | `AzureComputerVisionSecrets` | No       | `None`                 | Computer Vision endpoint          |
| `AZURE_CV_API_KEY`               | `AzureComputerVisionSecrets` | No       | `None`                 | Computer Vision API key           |
| `AZURE_CV_API_VERSION`           | `AzureComputerVisionSecrets` | No       | `2024-02-01`           | CV API version                    |
| `AZURE_CV_MODEL_VERSION`         | `AzureComputerVisionSecrets` | No       | `2023-04-15`           | Florence model version            |

**Runtime environment** (macOS): `SSL_CERT_FILE=/private/etc/ssl/cert.pem` — required for venv Python to access Azure HTTPS endpoints.

---

## Data Models

Defined in `src/ai_search/models.py`:

| Model                  | Key Fields                                                                        | Purpose                            |
|------------------------|-----------------------------------------------------------------------------------|------------------------------------|
| `CharacterDescription` | `character_id`, `semantic`, `emotion`, `pose`                                    | Per-character descriptions         |
| `ImageMetadata`        | `scene_type`, `time_of_day`, `lighting_condition`, `primary_subject`, `artistic_style`, `tags`, `narrative_theme` | Synthetic metadata                 |
| `NarrativeIntent`      | `story_summary`, `narrative_type`, `tone`                                        | Narrative analysis                 |
| `EmotionalTrajectory`  | `starting_emotion`, `mid_emotion`, `end_emotion`, `emotional_polarity`           | Emotion tracking [-1, 1]           |
| `RequiredObjects`      | `key_objects`, `contextual_objects`, `symbolic_elements`                          | Object detection                   |
| `LowLightMetrics`      | `brightness_score`, `contrast_score`, `noise_estimate`, `shadow_dominance`, `visibility_confidence` | Low-light quality (all 0.0–1.0)    |
| `ImageExtraction`      | All descriptions + `characters`, `metadata`, `narrative`, `emotion`, `objects`, `low_light` | Full GPT-4o extraction output      |
| `ImageVectors`         | `semantic_vector`, `structural_vector`, `style_vector`, `image_vector`           | All 4 embedding vectors            |
| `SearchDocument`       | 15 primitive + 4 vector fields                                                   | Index upload format                |
| `SearchResult`         | `image_id`, `search_score`, `generation_prompt`, `scene_type`, `tags`            | Query result format                |

---

## Project Structure

```text
ai-search/
├── config.yaml                          # Non-secret configuration
├── pyproject.toml                       # Dependencies, entry points, tool config
├── README.md                            # Setup and usage guide
├── requirements.md                      # Full requirements specification
├── data/
│   └── sample_images.json               # 10 sample images for testing
├── docs/
│   ├── architecture.md                  # This document
│   └── learnings-image-embedding-fix.md # Post-mortem and learnings (17 issues)
├── scripts/
│   ├── ingest_samples.py                # Batch ingestion of sample images
│   ├── test_search.py                   # Basic keyword/filter search test
│   ├── test_image_search.py             # Image similarity search test
│   ├── test_image_embed.py              # Smoke test for embed-v-4-0
│   ├── test_inference_sdk.py            # Azure AI Inference SDK test
│   ├── test_hybrid_vs_image.py          # Image-only vs hybrid search comparison
│   ├── test_relevance_tiers.py          # Tiered relevance demo
│   ├── test_relevance.py                # Relative relevance scoring demo
│   └── analyze_scores.py                # Score distribution analysis
├── src/ai_search/
│   ├── __init__.py                      # Package version
│   ├── clients.py                       # Client factories (all Azure services, cached)
│   ├── config.py                        # Pydantic config + secrets loading
│   ├── models.py                        # Shared data models
│   ├── embeddings/
│   │   ├── encoder.py                   # text-embedding-3-large wrapper
│   │   ├── image.py                     # Image embedding (Foundry + Florence backends)
│   │   ├── pipeline.py                  # Orchestrates all 4 vectors
│   │   ├── semantic.py                  # 3072d semantic embedding
│   │   ├── structural.py               # 1024d structural embedding
│   │   └── style.py                     # 512d style embedding
│   ├── extraction/
│   │   ├── extractor.py                 # GPT-4o unified extraction
│   │   ├── emotion.py                   # EmotionalTrajectory accessor
│   │   ├── low_light.py                 # LowLightMetrics accessor
│   │   ├── narrative.py                 # NarrativeIntent accessor
│   │   └── objects.py                   # RequiredObjects accessor
│   ├── indexing/
│   │   ├── schema.py                    # Index schema + HNSW config + scoring profile
│   │   ├── indexer.py                   # Document batch upload with retry
│   │   └── cli.py                       # Index management CLI
│   ├── ingestion/
│   │   ├── loader.py                    # ImageInput model + factory methods
│   │   ├── metadata.py                  # Synthetic metadata generation
│   │   └── cli.py                       # Ingestion CLI entry point
│   └── retrieval/
│       ├── query.py                     # Query vector generation (LLM expansion)
│       ├── search.py                    # Hybrid search execution (BM25 + RRF)
│       ├── relevance.py                 # Relative relevance scoring
│       ├── pipeline.py                  # Retrieval orchestration
│       └── cli.py                       # Query CLI entry point
└── tests/
    ├── conftest.py                      # Shared fixtures
    ├── test_config.py                   # Config loading tests
    ├── test_models.py                   # Data model tests
    ├── test_embeddings/                 # Encoder, image, pipeline tests
    ├── test_extraction/                 # Extractor tests
    ├── test_indexing/                   # Indexer, schema tests
    ├── test_integration/                # E2E integration tests
    └── test_retrieval/                  # Query, search tests
```

### Module Responsibilities

| Module                          | Responsibility                                            |
|---------------------------------|-----------------------------------------------------------|
| `ai_search.config`             | Pydantic config models, YAML loading, env var secrets     |
| `ai_search.clients`            | Factory functions for all Azure SDK clients (cached with `@lru_cache`) |
| `ai_search.models`             | Shared Pydantic data models used across the pipeline      |
| `ai_search.embeddings.encoder` | Base `embed_texts()` / `embed_text()` using text-embedding-3-large |
| `ai_search.embeddings.image`   | Multi-backend image embedding (Foundry + Florence)        |
| `ai_search.embeddings.pipeline` | Orchestrates all 4 embedding vectors with async parallelism |
| `ai_search.extraction.extractor` | Unified GPT-4o vision extraction → `ImageExtraction`   |
| `ai_search.indexing.schema`    | Azure AI Search index schema definition and creation      |
| `ai_search.indexing.indexer`   | Document batch upload with exponential backoff retry      |
| `ai_search.retrieval.query`    | Query vector generation with LLM expansion                |
| `ai_search.retrieval.search`   | Hybrid search execution (BM25 + multi-vector RRF)        |
| `ai_search.retrieval.relevance` | Relative relevance scoring with confidence tiers         |

---

## CLI Entry Points

Defined in `pyproject.toml`:

| Command              | Module                        | Purpose                    |
|----------------------|-------------------------------|----------------------------|
| `ai-search-ingest`  | `ai_search.ingestion.cli:main` | Ingest a single image      |
| `ai-search-index`   | `ai_search.indexing.cli:main`  | Index management (create/update) |
| `ai-search-query`   | `ai_search.retrieval.cli:main` | Execute search queries     |

### CLI Arguments

**`ai-search-ingest`**:

| Argument       | Required         | Description                    |
|----------------|------------------|--------------------------------|
| `--image-url`  | One of url/file  | URL of the image to ingest     |
| `--image-file` | One of url/file  | Local path to the image file   |
| `--prompt`     | **Yes**          | Generation prompt for the image |
| `--image-id`   | **Yes**          | Unique identifier              |

**`ai-search-index`**:

| Subcommand | Description                       |
|------------|-----------------------------------|
| `create`   | Create or update the search index |

**`ai-search-query`**:

| Argument   | Required | Default | Description                  |
|------------|----------|---------|------------------------------|
| `--query`  | **Yes**  | —       | Search query text            |
| `--top`    | No       | 10      | Number of results to return  |
| `--filter` | No       | `None`  | OData filter expression      |

---

## Future Implementation

The following advanced features are designed for Azure AI Search **Standard S1 tier** (~$250/month) which supports up to 100+ vector fields and semantic ranker.

### Character-Level Vector Embeddings

Per-character embeddings capture individual character attributes for fine-grained retrieval.

**Additional vector fields (9 fields, 3 slots × 3 types):**

| Field Pattern              | Dimensions | Source                    |
|----------------------------|------------|---------------------------|
| `char_{n}_semantic_vector` | 512        | Character semantic description |
| `char_{n}_emotion_vector`  | 256        | Character emotional state |
| `char_{n}_pose_vector`     | 256        | Character body pose/position |

Where `n` = 0, 1, 2 (configurable via `max_character_slots`).

**Implementation:**

* `CharacterVectors` model: `character_id`, `semantic_vector`, `emotion_vector`, `pose_vector`
* `ImageVectors.character_vectors: list[CharacterVectors]`
* `generate_character_vectors()`: parallel embedding of each character description
* Flattened into `SearchDocument` as `char_0_semantic_vector`, `char_1_semantic_vector`, etc.

**Index impact:** 4 current + 9 character = 13 vector fields (requires Standard tier).

### Semantic Ranker

Azure AI Search semantic ranker provides L2 re-ranking using a Microsoft-trained transformer model.

**Configuration:**

```python
SemanticConfiguration(
    name="default-semantic-config",
    prioritized_fields=SemanticPrioritizedFields(
        content_fields=[SemanticField(field_name="generation_prompt")],
    ),
)
```

**Cost:** ~$1 per 1,000 queries on top of Standard tier base.

### Three-Stage Retrieval Pipeline

Advanced retrieval with application-level re-ranking and diversity selection.

**Stage 1 — Hybrid Search** (current implementation, retained):

* BM25 + multi-vector RRF fusion
* Returns `stage1_top_k` candidates (e.g., 200)

**Stage 2 — Application Re-Ranking:**

Rule-based re-ranker using document metadata to compute a composite `rerank_score`:

```python
rerank_score = (
    weights.emotional * emotional_match_score
    + weights.narrative * narrative_match_score
    + weights.object_overlap * object_overlap_score
    + weights.low_light * low_light_match_score
)
```

Configurable via `RerankWeightsConfig`:

| Weight         | Default | Purpose                    |
|----------------|---------|----------------------------|
| Emotional      | 0.3     | Emotion polarity alignment |
| Narrative      | 0.25    | Narrative type/theme match |
| Object overlap | 0.25    | Required object presence   |
| Low light      | 0.2     | Brightness constraint match |

Requires `QueryContext` model with: `query_text`, `emotions`, `narrative_intent`, `required_objects`, `low_light_score`.

**Stage 3 — MMR Diversity Selection:**

Maximal Marginal Relevance balances relevance with diversity:

$$MMR = \lambda \cdot Relevance(d_i) - (1 - \lambda) \cdot \max_{d_j \in S} Similarity(d_i, d_j)$$

* `mmr_lambda`: 0.6 (configurable) — higher values favor relevance over diversity
* Uses semantic vectors for cosine similarity between candidates
* Returns `stage3_top_k` final results (e.g., 20)

**Retrieval config (future):**

```yaml
retrieval:
  stage1_top_k: 200
  stage1_k_nearest: 100
  stage2_top_k: 50
  stage3_top_k: 20
  mmr_lambda: 0.6
  rerank_weights:
    emotional: 0.3
    narrative: 0.25
    object_overlap: 0.25
    low_light: 0.2
```

### Migration Path

To upgrade from Basic to Standard tier with advanced features:

1. Upgrade Azure AI Search to Standard S1
2. Restore character embedding module (`embeddings/character.py`)
3. Add `CharacterVectors` model and `character_vectors` field to `ImageVectors`
4. Add 9 `char_*` vector fields to `SearchDocument`
5. Update `IndexConfig` with `max_character_slots: 3` and character dimension settings
6. Re-enable character vector loop in schema builder
7. Add `SemanticConfiguration` to index schema
8. Restore `reranker.py` and `diversity.py` modules
9. Update retrieval pipeline to three-stage flow
10. Re-index all documents to populate character vectors
