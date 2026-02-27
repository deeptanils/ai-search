# Architecture Details Research — ai-search

**Date**: 2026-02-27
**Scope**: Complete granular analysis of the `ai-search` codebase for architecture document update

---

## 1. API Endpoints and SDK Details

### 1.1 Azure Services Used

| Service | Purpose | Client(s) |
|---|---|---|
| Azure AI Foundry (OpenAI) | GPT-4o extraction, text-embedding-3-large | `AzureOpenAI`, `AsyncAzureOpenAI` |
| Azure AI Foundry (Inference) | embed-v-4-0 text + image embeddings | `EmbeddingsClient`, `ImageEmbeddingsClient` |
| Azure AI Search | Index management, document upload, hybrid search | `SearchIndexClient`, `SearchClient` |
| Azure Computer Vision 4.0 | Florence image/text vectorization (alt backend) | `httpx.AsyncClient` (REST) |

### 1.2 SDK Packages and Versions

From [pyproject.toml](pyproject.toml):

| Package | Version Constraint | Purpose |
|---|---|---|
| `openai` | `>=1.58.0` | Azure OpenAI client for GPT-4o and text-embedding-3-large |
| `azure-search-documents` | `>=11.6.0` | Azure AI Search index + document operations |
| `azure-identity` | `>=1.17.0` | `DefaultAzureCredential`, `get_bearer_token_provider` |
| `azure-ai-inference` | `>=1.0.0b9` | `EmbeddingsClient`, `ImageEmbeddingsClient` for Foundry models |
| `aiohttp` | `>=3.9` | Required by `azure-ai-inference` async clients |
| `pydantic` | `>=2.0` | Data models |
| `pydantic-settings` | `>=2.0` | Environment variable loading |
| `pyyaml` | `>=6.0` | `config.yaml` parsing |
| `python-dotenv` | `>=1.0` | `.env` file loading |
| `pillow` | `>=10.0` | Image resizing before embedding |
| `httpx` | `>=0.27` | Azure Computer Vision REST calls, image downloads |
| `structlog` | `>=24.0` | Structured logging |
| `numpy` | `>=1.26` | Numeric operations |

### 1.3 Specific API Calls Made

#### Azure OpenAI (via `openai` SDK)

| Method | Model | Location | Purpose |
|---|---|---|---|
| `client.beta.chat.completions.parse()` | `gpt-4o` | [src/ai_search/extraction/extractor.py](src/ai_search/extraction/extractor.py#L53) | Unified image extraction with structured output (`response_format=ImageExtraction`) |
| `client.beta.chat.completions.parse()` | `gpt-4o` | [src/ai_search/ingestion/metadata.py](src/ai_search/ingestion/metadata.py#L28) | Synthetic metadata generation (`response_format=ImageMetadata`) |
| `client.chat.completions.create()` | `gpt-4o` | [src/ai_search/retrieval/query.py](src/ai_search/retrieval/query.py#L39) | Structural/style description expansion for query vectors |
| `client.embeddings.create()` | `text-embedding-3-large` | [src/ai_search/embeddings/encoder.py](src/ai_search/embeddings/encoder.py#L32) | Text embedding at configurable dimensions |

#### Azure AI Inference (via `azure-ai-inference` SDK)

| Method | Model | Location | Purpose |
|---|---|---|---|
| `EmbeddingsClient.embed()` | `embed-v-4-0` | [src/ai_search/embeddings/image.py](src/ai_search/embeddings/image.py#L200) | Text embedding in image space (cross-modal) |
| `ImageEmbeddingsClient.embed()` | `embed-v-4-0` | [src/ai_search/embeddings/image.py](src/ai_search/embeddings/image.py#L185) | Image embedding via `ImageEmbeddingInput` |

**SDK classes imported** in [src/ai_search/clients.py](src/ai_search/clients.py#L9):

- `azure.ai.inference.aio.EmbeddingsClient`
- `azure.ai.inference.aio.ImageEmbeddingsClient`
- `azure.ai.inference.models.ImageEmbeddingInput`

#### Azure Computer Vision 4.0 (Florence — REST via `httpx`)

| Route | Location | Purpose |
|---|---|---|
| `POST /computervision/retrieval:vectorizeImage` | [src/ai_search/embeddings/image.py](src/ai_search/embeddings/image.py#L94) | Image-to-vector (1024d) |
| `POST /computervision/retrieval:vectorizeText` | [src/ai_search/embeddings/image.py](src/ai_search/embeddings/image.py#L113) | Text-to-vector in image space (1024d) |

Query parameters: `api-version` (default `2024-02-01`), `model-version` (default `2023-04-15`).

#### Azure AI Search (via `azure-search-documents` SDK)

| Method | Location | Purpose |
|---|---|---|
| `SearchIndexClient.create_or_update_index()` | [src/ai_search/indexing/schema.py](src/ai_search/indexing/schema.py#L123) | Create or update the search index |
| `SearchClient.upload_documents()` | [src/ai_search/indexing/indexer.py](src/ai_search/indexing/indexer.py#L76) | Batch document upload with retry |
| `SearchClient.search()` | [src/ai_search/retrieval/search.py](src/ai_search/retrieval/search.py#L104) | Hybrid search (BM25 + multi-vector RRF) |
| `SearchClient.get_document()` | [scripts/ingest_samples.py](scripts/ingest_samples.py#L157) | Check if document already indexed |

### 1.4 Authentication Methods

| Service | Auth Method | Details | Code Location |
|---|---|---|---|
| Azure OpenAI | **Entra ID** (`DefaultAzureCredential`) | Uses `get_bearer_token_provider` with scope `https://cognitiveservices.azure.com/.default` | [src/ai_search/clients.py](src/ai_search/clients.py#L37) |
| Azure AI Inference (Foundry) | **Entra ID** (`DefaultAzureCredential`) | `credential_scopes=["https://cognitiveservices.azure.com/.default"]` | [src/ai_search/clients.py](src/ai_search/clients.py#L99) |
| Azure AI Search | **API Key** (`AzureKeyCredential`) | Loaded from `AZURE_AI_SEARCH_API_KEY` env var | [src/ai_search/clients.py](src/ai_search/clients.py#L69) |
| Azure Computer Vision | **API Key** (HTTP header) | `Ocp-Apim-Subscription-Key` header from `AZURE_CV_API_KEY` | [src/ai_search/clients.py](src/ai_search/clients.py#L134) |

**Credential scope**: `https://cognitiveservices.azure.com/.default` — used for both OpenAI and Foundry Inference clients.

---

## 2. Configuration Details

### 2.1 Full config.yaml

Source: [config.yaml](config.yaml)

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

### 2.2 Environment Variables

All loaded via `pydantic-settings` with `SettingsConfigDict(env_file=".env")`.

Source: [src/ai_search/config.py](src/ai_search/config.py)

| Variable | Prefix Class | Required | Default | Purpose |
|---|---|---|---|---|
| `AZURE_FOUNDRY_ENDPOINT` | `AzureFoundrySecrets` | **Yes** | — | Azure AI Foundry base endpoint |
| `AZURE_FOUNDRY_EMBED_ENDPOINT` | `AzureFoundrySecrets` | No | `None` | Foundry models endpoint for Inference SDK |
| `AZURE_FOUNDRY_API_KEY` | `AzureFoundrySecrets` | No | `None` | API key (fallback, Entra ID preferred) |
| `AZURE_OPENAI_API_VERSION` | `AzureOpenAISecrets` | No | `2024-12-01-preview` | OpenAI API version |
| `AZURE_AI_SEARCH_ENDPOINT` | `AzureSearchSecrets` | **Yes** | — | Azure AI Search endpoint |
| `AZURE_AI_SEARCH_API_KEY` | `AzureSearchSecrets` | **Yes** | — | Azure AI Search admin API key |
| `AZURE_AI_SEARCH_INDEX_NAME` | `AzureSearchSecrets` | No | `candidate-index` | Index name |
| `AZURE_CV_ENDPOINT` | `AzureComputerVisionSecrets` | No | `None` | Computer Vision endpoint |
| `AZURE_CV_API_KEY` | `AzureComputerVisionSecrets` | No | `None` | Computer Vision API key |
| `AZURE_CV_API_VERSION` | `AzureComputerVisionSecrets` | No | `2024-02-01` | CV API version |
| `AZURE_CV_MODEL_VERSION` | `AzureComputerVisionSecrets` | No | `2023-04-15` | Florence model version |

### 2.3 Pydantic Config Models

Source: [src/ai_search/config.py](src/ai_search/config.py)

```
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

Secrets classes (all `BaseSettings` with `SettingsConfigDict`):

- `AzureFoundrySecrets` — prefix `AZURE_FOUNDRY_`
- `AzureOpenAISecrets` — prefix `AZURE_OPENAI_`
- `AzureSearchSecrets` — prefix `AZURE_AI_SEARCH_`
- `AzureComputerVisionSecrets` — prefix `AZURE_CV_`

All config loading functions use `@lru_cache(maxsize=1)`.

---

## 3. Embedding Pipeline Granular Details

### 3.1 Text Embeddings (text-embedding-3-large)

Source: [src/ai_search/embeddings/encoder.py](src/ai_search/embeddings/encoder.py)

- **SDK class**: `openai.AsyncAzureOpenAI` (via `get_async_openai_client()`)
- **Method**: `client.embeddings.create(model=..., input=..., dimensions=...)`
- **Model**: `text-embedding-3-large` (supports Matryoshka dimensions)
- **Batching**: Chunks input by `embedding_chunk_size` (default 2048)
- **Input format**: `list[str]` — plain text strings
- **Output format**: `list[list[float]]` — one vector per input text

Three specialized wrappers call `embed_text()` with different dimensions:

| Module | Dimension | Source text |
|---|---|---|
| [semantic.py](src/ai_search/embeddings/semantic.py) | 3072 | `extraction.semantic_description` |
| [structural.py](src/ai_search/embeddings/structural.py) | 1024 | `extraction.structural_description` |
| [style.py](src/ai_search/embeddings/style.py) | 512 | `extraction.style_description` |

### 3.2 Image Embeddings (embed-v-4-0 / Florence)

Source: [src/ai_search/embeddings/image.py](src/ai_search/embeddings/image.py)

**Backend selection**: Controlled by `config.models.image_embedding_model`:

- `"azure-cv-florence"` → Florence backend (Computer Vision REST)
- Any other value (e.g., `"embed-v-4-0"`) → Foundry backend (AI Inference SDK)

#### Foundry Backend (default: embed-v-4-0)

- **Image SDK class**: `azure.ai.inference.aio.ImageEmbeddingsClient`
- **Text SDK class**: `azure.ai.inference.aio.EmbeddingsClient`
- **Image input**: `ImageEmbeddingInput(image=data_uri)` — base64 data URI
- **Text input**: `list[str]` — plain text for cross-modal search
- **Method (image)**: `client.embed(input=[ImageEmbeddingInput(...)], model=..., dimensions=...)`
- **Method (text)**: `client.embed(input=[text], model=..., dimensions=...)`
- **Dimensions**: Configurable, default 1024

#### Florence Backend (Azure Computer Vision 4.0)

- **Client**: `httpx.AsyncClient` with `base_url=secrets.endpoint`
- **Auth header**: `Ocp-Apim-Subscription-Key`
- **Image route**: `POST /computervision/retrieval:vectorizeImage`
  - JSON body: `{"url": image_url}` (URL input)
  - Binary body with `Content-Type: application/octet-stream` (bytes input)
- **Text route**: `POST /computervision/retrieval:vectorizeText`
  - JSON body: `{"text": query_text}`
- **Fixed dimensions**: 1024 (not configurable)

### 3.3 Image Preprocessing

Source: [src/ai_search/embeddings/image.py](src/ai_search/embeddings/image.py#L141) — `_resize_image_bytes()`

- **Max size**: 512×512 pixels (constant `_MAX_IMAGE_SIZE = 512`)
- **Algorithm**: `Image.LANCZOS` (Pillow)
- **Color mode**: Converted to `RGB` if not already
- **Output format**: JPEG, quality 80
- **Purpose**: Reduce base64 payload size, stay within S0-tier rate limits

### 3.4 Embedding Pipeline Orchestration

Source: [src/ai_search/embeddings/pipeline.py](src/ai_search/embeddings/pipeline.py)

**`generate_all_vectors(extraction, image_url, image_bytes) -> ImageVectors`**

1. **Step 1**: Text embeddings run in **parallel** via `asyncio.gather()`:
   - `generate_semantic_vector()` — 3072d
   - `generate_structural_vector()` — 1024d
   - `generate_style_vector()` — 512d
2. **Step 2**: Image embedding runs **separately** (rate-limited on S0 tier):
   - `embed_image()` — 1024d (only if `image_url` or `image_bytes` provided)

### 3.5 Vector Validation

Source: [src/ai_search/embeddings/image.py](src/ai_search/embeddings/image.py#L42) — `_validate_vector()`

- Checks vector is not `None` or empty
- Checks vector length matches `expected_dims`
- Raises `ValueError` with diagnostic info on mismatch
- Logs warning with `expected_dimensions`, `actual_dimensions`, `response_keys`

---

## 4. Index Schema Details

### 4.1 Full Schema Definition

Source: [src/ai_search/indexing/schema.py](src/ai_search/indexing/schema.py)

#### HNSW Configuration

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

Profile name: `hnsw-cosine-profile`

#### Vector Search Configuration

```python
VectorSearch(
    algorithms=[hnsw_algo],
    profiles=[
        VectorSearchProfile(
            name="hnsw-cosine-profile",
            algorithm_configuration_name="hnsw-cosine",
        ),
    ],
)
```

#### Field Definitions

| Field | SDK Class | Type | Key | Filterable | Sortable | Facetable | Searchable |
|---|---|---|---|---|---|---|---|
| `image_id` | `SimpleField` | `String` | **Yes** | Yes | — | — | — |
| `generation_prompt` | `SearchableField` | `String` | — | — | — | — | Yes |
| `scene_type` | `SimpleField` | `String` | — | Yes | — | Yes | — |
| `time_of_day` | `SimpleField` | `String` | — | Yes | — | — | — |
| `lighting_condition` | `SimpleField` | `String` | — | Yes | — | Yes | — |
| `primary_subject` | `SimpleField` | `String` | — | Yes | — | — | — |
| `artistic_style` | `SimpleField` | `String` | — | Yes | — | Yes | — |
| `tags` | `SearchField` | `Collection(String)` | — | Yes | — | Yes | Yes |
| `narrative_theme` | `SimpleField` | `String` | — | Yes | — | — | — |
| `narrative_type` | `SimpleField` | `String` | — | Yes | — | — | — |
| `emotional_polarity` | `SimpleField` | `Double` | — | Yes | Yes | — | — |
| `low_light_score` | `SimpleField` | `Double` | — | Yes | — | — | — |
| `character_count` | `SimpleField` | `Int32` | — | Yes | Yes | — | — |
| `metadata_json` | `SimpleField` | `String` | — | — | — | — | — |
| `extraction_json` | `SimpleField` | `String` | — | — | — | — | — |
| `semantic_vector` | `SearchField` | `Collection(Single)` | — | — | — | — | Yes (vector) |
| `structural_vector` | `SearchField` | `Collection(Single)` | — | — | — | — | Yes (vector) |
| `style_vector` | `SearchField` | `Collection(Single)` | — | — | — | — | Yes (vector) |
| `image_vector` | `SearchField` | `Collection(Single)` | — | — | — | — | Yes (vector) |

**Note**: The `tags` field uses `SearchField` (not `SearchableField`) to correctly handle `Collection(String)` type. See [learnings doc](docs/learnings-image-embedding-fix.md) learning #12.

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

Set as `default_scoring_profile="text-boost"`.

#### SDK Classes Used

From `azure.search.documents.indexes.models`:

- `HnswAlgorithmConfiguration`
- `HnswParameters`
- `ScoringProfile`
- `SearchableField`
- `SearchField`
- `SearchFieldDataType`
- `SearchIndex`
- `SimpleField`
- `TextWeights`
- `VectorSearch`
- `VectorSearchProfile`

---

## 5. Ingestion Pipeline

### 5.1 Image Processing Flow

Source: [src/ai_search/ingestion/cli.py](src/ai_search/ingestion/cli.py#L18) — `_process_image()`

1. **Extraction**: `extract_image(image_input)` — GPT-4o structured output
2. **Embedding**: `generate_all_vectors(extraction)` — 4 parallel/sequential vectors
3. **Document build**: `build_search_document(image_input, extraction, vectors)`
4. **Upload**: `upload_documents([doc])`

### 5.2 Image Input Handling

Source: [src/ai_search/ingestion/loader.py](src/ai_search/ingestion/loader.py)

`ImageInput` Pydantic model:

- `image_id: str`
- `generation_prompt: str`
- `image_url: str | None`
- `image_base64: str | None`

Factory methods:

- `ImageInput.from_url(image_id, prompt, url)` — URL-based input
- `ImageInput.from_file(image_id, prompt, path)` — Local file, auto base64-encodes

`to_openai_image_content()` — Converts to OpenAI vision content format:

```python
{"type": "image_url", "image_url": {"url": ..., "detail": "high"}}
```

### 5.3 Batch Ingestion

Source: [scripts/ingest_samples.py](scripts/ingest_samples.py)

| Parameter | Value | Purpose |
|---|---|---|
| `INTER_IMAGE_DELAY_S` | 10 seconds | Delay between images for rate limiting |
| `MAX_RETRIES` | 5 | Retry attempts on 429 errors |
| `RETRY_BACKOFF_S` | 30 seconds × attempt | Backoff multiplier |
| Batch size (upload) | 500 (from config) | Documents per upload batch |

**Process**:

1. Load sample images from [data/sample_images.json](data/sample_images.json)
2. **Pre-download** all images synchronously (`httpx.Client`) to avoid async SSL issues
3. Check which docs are already indexed (via `get_document()`) unless `--force` flag
4. Process each image: extract → embed → build doc
5. Upload one document at a time
6. Sleep `INTER_IMAGE_DELAY_S` between images
7. On 429 HTTP error: retry with `RETRY_BACKOFF_S × attempt_number` backoff

### 5.4 Document Upload with Retry

Source: [src/ai_search/indexing/indexer.py](src/ai_search/indexing/indexer.py)

`upload_documents(documents, max_retries=3, base_delay=1.0)`:

- Batches documents by `index_batch_size` (default 500)
- Calls `client.upload_documents(documents=docs)`
- On `HttpResponseError` with status 429 or 503: exponential backoff `base_delay * 2^attempt`
- Empty vector fields (`list` with `len == 0`) are omitted from the upload payload

### 5.5 Sample Data

Source: [data/sample_images.json](data/sample_images.json)

10 sample images from Unsplash with themed generation prompts:

| ID | Theme |
|---|---|
| sample-001 | Cinematic Tokyo night scene |
| sample-002 | Golden hour fisherman portrait |
| sample-003 | Aerial ocean waves on volcanic rocks |
| sample-004 | Abandoned industrial warehouse |
| sample-005 | Child in sunflower field |
| sample-006 | Foggy redwood forest |
| sample-007 | Macro dew drops on spider web |
| sample-008 | Jazz musicians on stage |
| sample-009 | Snow-covered mountain at dawn |
| sample-010 | Bangkok street food market |

---

## 6. Retrieval Pipeline

### 6.1 Query Vector Generation

Source: [src/ai_search/retrieval/query.py](src/ai_search/retrieval/query.py)

**`generate_query_vectors(query_text, query_image_url) -> dict[str, list[float]]`**

1. **LLM expansion** for structural + style descriptions:
   - `client.chat.completions.create(model="gpt-4o", temperature=0.2, max_tokens=200)`
   - Uses fixed system prompts (`STRUCTURAL_PROMPT`, `STYLE_PROMPT`)
2. **Parallel text embedding** via `asyncio.gather()`:
   - Semantic: raw query → 3072d
   - Structural: LLM-expanded → 1024d
   - Style: LLM-expanded → 512d
3. **Image-space vector**:
   - If `query_image_url`: `embed_image(image_url=...)` — visual image embedding
   - Otherwise: `embed_text_for_image_search(query_text)` — cross-modal text-to-image

Returns dict with keys: `semantic_vector`, `structural_vector`, `style_vector`, `image_vector`.

### 6.2 Hybrid Search Execution

Source: [src/ai_search/retrieval/search.py](src/ai_search/retrieval/search.py)

**`execute_hybrid_search(query_text, query_vectors, odata_filter, top, min_confidence)`**

Vector query construction:

```python
VectorizedQuery(
    vector=query_vectors["semantic_vector"],
    k_nearest_neighbors=100,      # from config.retrieval.k_nearest
    fields="semantic_vector",
    weight=0.4 * 10,              # config weight × 10 = 4.0
)
```

**Weight scaling**: Config weights are multiplied by 10 to preserve ratio versus BM25's implicit weight of 1.0.

| Vector field | Config weight | Effective weight (×10) |
|---|---|---|
| `semantic_vector` | 0.4 | 4.0 |
| `structural_vector` | 0.15 | 1.5 |
| `style_vector` | 0.15 | 1.5 |
| `image_vector` | 0.2 | 2.0 |
| BM25 text | 0.1 | implicit 1.0 |

**Search call**:

```python
client.search(
    search_text=query_text,
    vector_queries=vector_queries,
    filter=odata_filter,
    select=SELECT_FIELDS,
    top=search_top,            # default 50
)
```

**SELECT_FIELDS**: `image_id`, `generation_prompt`, `scene_type`, `tags`, `narrative_type`, `emotional_polarity`, `low_light_score`, `character_count`, `extraction_json`, `metadata_json`

### 6.3 Relevance Scoring

Source: [src/ai_search/retrieval/relevance.py](src/ai_search/retrieval/relevance.py)

**Problem solved**: Cosine similarity with high-dimensional embeddings (embed-v-4-0, 1024d) produces uniformly high scores (0.95+). Absolute thresholds cannot distinguish true matches from noise.

**Solution**: Relative metrics across the result set:

| Metric | Formula | Purpose |
|---|---|---|
| `gap` | `top1 - top2` | Absolute score gap |
| `gap_ratio` | `gap / top1` | Relative separation from runner-up |
| `z_score` | `(top1 - mean) / stdev` | Statistical outlier detection |
| `spread` | `max - min` | Score range across results |

**Confidence tiers**:

| Tier | z_score | gap_ratio | spread | Meaning |
|---|---|---|---|---|
| **HIGH** | ≥ 2.0 | ≥ 0.01 | ≥ 0.02 | Clear outlier, likely true match |
| **MEDIUM** | ≥ 1.3 | ≥ 0.005 | ≥ 0.015 | Moderately distinct, probable match |
| **LOW** | below thresholds | — | — | No confident match |

**`RelevanceResult`** dataclass fields: `confidence`, `top_score`, `gap`, `gap_ratio`, `z_score`, `spread`, `mean`, `stdev`

**`filter_by_relevance(documents, score_key, min_confidence)`**: Returns `(filtered_docs, RelevanceResult)`. Empty list if confidence is below `min_confidence`.

### 6.4 Retrieval Pipeline Orchestration

Source: [src/ai_search/retrieval/pipeline.py](src/ai_search/retrieval/pipeline.py)

**`retrieve(query_text, odata_filter, top) -> list[SearchResult]`**

1. `generate_query_vectors(query_text)` — 4 query vectors
2. `execute_hybrid_search(...)` — BM25 + multi-vector RRF
3. Map results to `SearchResult` models

### 6.5 Result Format

Source: [src/ai_search/models.py](src/ai_search/models.py#L89)

```python
class SearchResult(BaseModel):
    image_id: str
    search_score: float
    generation_prompt: str | None = None
    scene_type: str | None = None
    tags: list[str] = Field(default_factory=list)
```

---

## 7. Project Structure

### 7.1 Full Directory Tree

```
ai-search/
├── config.yaml                          # Non-secret configuration
├── pyproject.toml                       # Dependencies, entry points, tool config
├── README.md                            # Setup and usage guide
├── requirements.md                      # Full requirements specification
├── data/
│   └── sample_images.json               # 10 sample images for testing
├── docs/
│   ├── architecture.md                  # Architecture document
│   └── learnings-image-embedding-fix.md # Post-mortem and learnings
├── scripts/
│   ├── analyze_scores.py                # Score distribution analysis
│   ├── ingest_samples.py                # Batch ingestion of sample images
│   ├── test_hybrid_vs_image.py          # Image-only vs hybrid search comparison
│   ├── test_image_embed.py              # Smoke test for embed-v-4-0
│   ├── test_image_search.py             # Image similarity search test
│   ├── test_inference_sdk.py            # Azure AI Inference SDK test
│   ├── test_relevance_tiers.py          # Tiered relevance demo
│   ├── test_relevance.py               # Relative relevance scoring demo
│   └── test_search.py                  # Basic keyword/filter search test
├── src/
│   └── ai_search/
│       ├── __init__.py                  # Package version
│       ├── clients.py                   # Client factories for all Azure services
│       ├── config.py                    # Pydantic config + secret loading
│       ├── models.py                    # Shared data models
│       ├── embeddings/
│       │   ├── __init__.py              # Module docstring
│       │   ├── encoder.py              # text-embedding-3-large wrapper
│       │   ├── image.py                # Image embedding (Foundry + Florence)
│       │   ├── pipeline.py            # Orchestrates all 4 vectors
│       │   ├── semantic.py            # 3072d semantic embedding
│       │   ├── structural.py          # 1024d structural embedding
│       │   └── style.py               # 512d style embedding
│       ├── extraction/
│       │   ├── __init__.py              # Module docstring
│       │   ├── emotion.py              # Emotional trajectory accessor
│       │   ├── extractor.py            # GPT-4o unified extraction
│       │   ├── low_light.py            # Low-light metrics accessor
│       │   ├── narrative.py            # Narrative intent accessor
│       │   └── objects.py              # Required objects accessor
│       ├── indexing/
│       │   ├── __init__.py              # Module docstring
│       │   ├── cli.py                  # Index management CLI
│       │   ├── indexer.py              # Document upload with retry
│       │   └── schema.py              # Index schema + HNSW config
│       ├── ingestion/
│       │   ├── __init__.py              # Module docstring
│       │   ├── cli.py                  # Ingestion CLI entry point
│       │   ├── loader.py              # ImageInput model + factory methods
│       │   └── metadata.py            # Synthetic metadata generation
│       └── retrieval/
│           ├── __init__.py              # Module docstring
│           ├── cli.py                  # Query CLI entry point
│           ├── pipeline.py            # Retrieval orchestration
│           ├── query.py               # Query vector generation
│           ├── relevance.py           # Relative relevance scoring
│           └── search.py             # Hybrid search execution
└── tests/
    ├── conftest.py                     # Shared fixtures
    ├── test_config.py                  # Config loading tests
    ├── test_models.py                  # Data model tests
    ├── test_embeddings/
    │   ├── test_encoder.py             # Encoder tests
    │   ├── test_image.py              # Image embedding tests
    │   └── test_pipeline.py           # Pipeline orchestration tests
    ├── test_extraction/
    │   └── test_extractor.py          # Extractor tests
    ├── test_indexing/
    │   ├── test_indexer.py            # Upload tests
    │   └── test_schema.py            # Schema tests
    ├── test_integration/
    │   ├── conftest.py                # Integration fixtures
    │   └── test_end_to_end.py        # E2E integration tests
    └── test_retrieval/
        ├── test_query.py              # Query vector tests
        └── test_search.py            # Search tests
```

### 7.2 Module Responsibilities

| Module | Responsibility |
|---|---|
| `ai_search.config` | Pydantic config models, YAML loading, env var secrets |
| `ai_search.clients` | Factory functions for all Azure SDK clients (cached) |
| `ai_search.models` | Shared Pydantic data models pipeline-wide |
| `ai_search.embeddings.encoder` | Base `embed_texts()` / `embed_text()` using text-embedding-3-large |
| `ai_search.embeddings.image` | Multi-backend image embedding (Foundry + Florence) |
| `ai_search.embeddings.pipeline` | Orchestrates all 4 embedding vectors |
| `ai_search.embeddings.semantic` | 3072d semantic vector generation |
| `ai_search.embeddings.structural` | 1024d structural vector generation |
| `ai_search.embeddings.style` | 512d style vector generation |
| `ai_search.extraction.extractor` | Unified GPT-4o vision extraction → `ImageExtraction` |
| `ai_search.extraction.emotion` | Accessor for `EmotionalTrajectory` from extraction |
| `ai_search.extraction.narrative` | Accessor for `NarrativeIntent` from extraction |
| `ai_search.extraction.objects` | Accessor for `RequiredObjects` from extraction |
| `ai_search.extraction.low_light` | Accessor for `LowLightMetrics` from extraction |
| `ai_search.indexing.schema` | Azure AI Search index schema definition |
| `ai_search.indexing.indexer` | Document batch upload with retry |
| `ai_search.indexing.cli` | CLI for index create/update |
| `ai_search.ingestion.loader` | `ImageInput` model and factory methods |
| `ai_search.ingestion.metadata` | GPT-4o metadata generation |
| `ai_search.ingestion.cli` | CLI for single-image ingestion |
| `ai_search.retrieval.query` | Query vector generation with LLM expansion |
| `ai_search.retrieval.search` | Hybrid search execution (BM25 + multi-vector RRF) |
| `ai_search.retrieval.relevance` | Relative relevance scoring with confidence tiers |
| `ai_search.retrieval.pipeline` | Retrieval orchestration |
| `ai_search.retrieval.cli` | CLI for query execution |

---

## 8. Scripts and Entry Points

### 8.1 CLI Entry Points

Defined in [pyproject.toml](pyproject.toml#L37):

| Command | Module | Purpose |
|---|---|---|
| `ai-search-ingest` | `ai_search.ingestion.cli:main` | Ingest a single image |
| `ai-search-index` | `ai_search.indexing.cli:main` | Index management (create/update) |
| `ai-search-query` | `ai_search.retrieval.cli:main` | Execute search queries |

### 8.2 CLI Arguments

#### `ai-search-ingest`

| Argument | Required | Description |
|---|---|---|
| `--image-url` | One of url/file | URL of the image to ingest |
| `--image-file` | One of url/file | Local path to the image file |
| `--prompt` | **Yes** | Generation prompt for the image |
| `--image-id` | **Yes** | Unique identifier for the image |

#### `ai-search-index`

| Subcommand | Description |
|---|---|
| `create` | Create or update the search index |

#### `ai-search-query`

| Argument | Required | Default | Description |
|---|---|---|---|
| `--query` | **Yes** | — | Search query text |
| `--top` | No | 10 | Number of results to return |
| `--filter` | No | `None` | OData filter expression |

### 8.3 Runnable Scripts

All scripts use `PYTHONPATH=src` and many require `SSL_CERT_FILE=/private/etc/ssl/cert.pem` on macOS.

| Script | Required Env Vars | Purpose |
|---|---|---|
| [scripts/ingest_samples.py](scripts/ingest_samples.py) | All Foundry + Search vars | Batch ingest 10 sample images. Args: `--dry-run`, `--force` |
| [scripts/test_search.py](scripts/test_search.py) | Search vars | Basic keyword/filter search verification |
| [scripts/test_image_search.py](scripts/test_image_search.py) | Foundry + Search vars | Image-to-image similarity search. Arg: `[url]` |
| [scripts/test_image_embed.py](scripts/test_image_embed.py) | Foundry vars | Smoke test embed-v-4-0 text + image embedding |
| [scripts/test_inference_sdk.py](scripts/test_inference_sdk.py) | Foundry vars | Live test of `EmbeddingsClient.embed()` |
| [scripts/test_hybrid_vs_image.py](scripts/test_hybrid_vs_image.py) | Foundry + Search vars | Compare image-only vs hybrid search relevance |
| [scripts/test_relevance_tiers.py](scripts/test_relevance_tiers.py) | Foundry + Search vars | Demonstrate HIGH/MEDIUM/LOW confidence tiers |
| [scripts/test_relevance.py](scripts/test_relevance.py) | Foundry + Search vars | Relative relevance scoring with metrics |
| [scripts/analyze_scores.py](scripts/analyze_scores.py) | Foundry + Search vars | Analyze why vector search scores are uniformly high |

### 8.4 Required Environment Variables by Group

**Foundry vars** (for OpenAI + Foundry Inference):

- `AZURE_FOUNDRY_ENDPOINT`
- `AZURE_FOUNDRY_EMBED_ENDPOINT`
- `AZURE_FOUNDRY_API_KEY` (optional if Entra ID configured)

**Search vars**:

- `AZURE_AI_SEARCH_ENDPOINT`
- `AZURE_AI_SEARCH_API_KEY`

**CV vars** (only if using Florence backend):

- `AZURE_CV_ENDPOINT`
- `AZURE_CV_API_KEY`

**Runtime**:

- `SSL_CERT_FILE=/private/etc/ssl/cert.pem` (macOS venv workaround)

---

## 9. Data Models Summary

Source: [src/ai_search/models.py](src/ai_search/models.py)

| Model | Fields | Purpose |
|---|---|---|
| `CharacterDescription` | `character_id`, `semantic`, `emotion`, `pose` | Per-character descriptions |
| `ImageMetadata` | `scene_type`, `time_of_day`, `lighting_condition`, `primary_subject`, `secondary_subjects`, `artistic_style`, `color_palette`, `tags`, `narrative_theme` | Synthetic metadata |
| `NarrativeIntent` | `story_summary`, `narrative_type`, `tone` | Narrative analysis |
| `EmotionalTrajectory` | `starting_emotion`, `mid_emotion`, `end_emotion`, `emotional_polarity` | Emotion tracking |
| `RequiredObjects` | `key_objects`, `contextual_objects`, `symbolic_elements` | Object detection |
| `LowLightMetrics` | `brightness_score`, `contrast_score`, `noise_estimate`, `shadow_dominance`, `visibility_confidence` | Low-light quality (all 0.0–1.0) |
| `ImageExtraction` | `semantic_description`, `structural_description`, `style_description`, `characters`, `metadata`, `narrative`, `emotion`, `objects`, `low_light` | Full GPT-4o extraction output |
| `ImageVectors` | `semantic_vector`, `structural_vector`, `style_vector`, `image_vector` | All embedding vectors |
| `SearchDocument` | 15 primitive + 4 vector fields | Index upload format |
| `SearchResult` | `image_id`, `search_score`, `generation_prompt`, `scene_type`, `tags` | Query result format |

---

## 10. GPT-4o Extraction Details

### 10.1 Extraction System Prompt

Source: [src/ai_search/extraction/extractor.py](src/ai_search/extraction/extractor.py#L15) — `EXTRACTION_SYSTEM_PROMPT`

Instructs GPT-4o to produce:

- `semantic_description`: 200-word rich description (scene, subjects, mood, themes)
- `structural_description`: 150-word spatial/composition analysis
- `style_description`: 150-word artistic style analysis
- Per-character `semantic`, `emotion`, `pose` descriptions (2–3 sentences each)
- `metadata`: Full `ImageMetadata` structure
- `narrative`: Story summary, type, tone
- `emotion`: Starting/mid/end emotions, polarity
- `objects`: Key, contextual, symbolic
- `low_light`: 5 scores (0.0–1.0)

### 10.2 Extraction API Call

```python
client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[system_prompt, user_content_with_image],
    response_format=ImageExtraction,   # Pydantic model as structured output
    temperature=0.2,
    max_tokens=4096,
)
```

### 10.3 Metadata Generation (Separate)

Source: [src/ai_search/ingestion/metadata.py](src/ai_search/ingestion/metadata.py)

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

## 11. Key Implementation Patterns

### 11.1 Client Caching

All client factories in [src/ai_search/clients.py](src/ai_search/clients.py) use `@lru_cache(maxsize=1)` for singleton behavior:

- `get_openai_client()` → `AzureOpenAI`
- `get_async_openai_client()` → `AsyncAzureOpenAI`
- `get_search_index_client()` → `SearchIndexClient`
- `get_foundry_embed_client()` → `EmbeddingsClient`
- `get_foundry_image_embed_client()` → `ImageEmbeddingsClient`
- `get_cv_client()` → `httpx.AsyncClient`

`get_search_client(index_name)` is **not** cached (allows different index names).

### 11.2 Config Loading

All config loaders in [src/ai_search/config.py](src/ai_search/config.py) use `@lru_cache(maxsize=1)`:

- `load_config()` — from `config.yaml`
- `load_foundry_secrets()` — from env vars
- `load_openai_secrets()` — from env vars
- `load_search_secrets()` — from env vars
- `load_cv_secrets()` — from env vars

### 11.3 Async/Sync Wrappers

Several modules provide sync wrappers using `asyncio.run()`:

- `embed_text_sync()` in [encoder.py](src/ai_search/embeddings/encoder.py#L52)
- `generate_query_vectors_sync()` in [query.py](src/ai_search/retrieval/query.py#L83)
- `retrieve_sync()` in [pipeline.py](src/ai_search/retrieval/pipeline.py#L53)

### 11.4 Error Handling and Retry

- **Embedding retry**: [scripts/ingest_samples.py](scripts/ingest_samples.py#L86) — 5 retries with 30s × attempt backoff on 429
- **Upload retry**: [src/ai_search/indexing/indexer.py](src/ai_search/indexing/indexer.py#L72) — 3 retries with exponential backoff (`1s × 2^attempt`) on 429/503
- **Vector validation**: [src/ai_search/embeddings/image.py](src/ai_search/embeddings/image.py#L42) — raises `ValueError` on dimension mismatch
