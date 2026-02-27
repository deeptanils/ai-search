<!-- markdownlint-disable-file -->
# Implementation Details: Multimodal Image Embeddings & Index Enhancements

## Context Reference

Sources:
* `.copilot-tracking/research/2026-02-26/model-strategy-index-config-research.md` — Florence API patterns (Lines 90-130), Cohere patterns (Lines 200-240), architecture recommendation (Lines 300-340)
* `src/ai_search/config.py` (Lines 1-172) — Current config structure
* `src/ai_search/clients.py` (Lines 1-68) — Current client factories
* `src/ai_search/embeddings/pipeline.py` (Lines 1-31) — Current embedding orchestrator

## Implementation Phase 1: Configuration & Secrets

<!-- parallelizable: false -->

### Step 1.1: Add Florence secrets model and config extensions to config.py

Add a new `AzureComputerVisionSecrets` pydantic-settings class and extend `ModelsConfig`, `VectorDimensionsConfig`, and `SearchWeightsConfig`.

Files:
* `src/ai_search/config.py` — Add `AzureComputerVisionSecrets`, extend `ModelsConfig`, `VectorDimensionsConfig`, `SearchWeightsConfig`

Changes to `config.py`:

1. Add `AzureComputerVisionSecrets` class after `AzureSearchSecrets` (after Line 55):

```python
class AzureComputerVisionSecrets(BaseSettings):
    """Azure Computer Vision (Florence) secrets from .env."""

    model_config = SettingsConfigDict(
        env_prefix="AZURE_CV_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    endpoint: str = ""
    api_key: str = ""
    api_version: str = "2024-02-01"
```

2. Add `image_embedding_model` to `ModelsConfig` (Line 62):

```python
class ModelsConfig(BaseModel):
    """Model deployment names."""

    embedding_model: str = "text-embedding-3-large"
    llm_model: str = "gpt-4o"
    image_embedding_model: str = "azure-cv-florence"
```

3. Add `image` dimension to `VectorDimensionsConfig` (Line 82):

```python
class VectorDimensionsConfig(BaseModel):
    """Vector dimension configuration."""

    semantic: int = 3072
    structural: int = 1024
    style: int = 512
    image: int = 1024  # Florence fixed dimension
    character_semantic: int = 512
    character_emotion: int = 256
    character_pose: int = 256
```

4. Add `image_weight` to `SearchWeightsConfig` (Line 72):

```python
class SearchWeightsConfig(BaseModel):
    """Retrieval weight configuration."""

    semantic_weight: float = 0.4
    structural_weight: float = 0.15
    style_weight: float = 0.15
    image_weight: float = 0.2
    keyword_weight: float = 0.1
```

Note: Weights redistributed to sum to 1.0 with the new `image_weight`.

5. Add `load_cv_secrets()` loader function (after `load_search_secrets`):

```python
@lru_cache(maxsize=1)
def load_cv_secrets() -> AzureComputerVisionSecrets:
    """Load Azure Computer Vision secrets from .env."""
    return AzureComputerVisionSecrets()
```

Success criteria:
* `AzureComputerVisionSecrets` loads from `AZURE_CV_` prefixed env vars
* `ModelsConfig.image_embedding_model` defaults to `"azure-cv-florence"`
* `VectorDimensionsConfig.image` defaults to `1024`
* `SearchWeightsConfig.image_weight` defaults to `0.2`
* All weights sum to 1.0

Context references:
* `src/ai_search/config.py` (Lines 13-55) — Existing secrets classes pattern
* Research doc (Lines 380-410) — Config changes needed

Dependencies:
* None — this is the foundation step

### Step 1.2: Update config.yaml with image embedding settings

Add image embedding model, image dimension, and image weight to `config.yaml`.

Files:
* `config.yaml` — Add `image_embedding_model`, `image` dimension, `image_weight`

Changes:

```yaml
models:
  embedding_model: text-embedding-3-large
  llm_model: gpt-4o
  image_embedding_model: azure-cv-florence

search:
  semantic_weight: 0.4
  structural_weight: 0.15
  style_weight: 0.15
  image_weight: 0.2
  keyword_weight: 0.1

index:
  vector_dimensions:
    semantic: 3072
    structural: 1024
    style: 512
    image: 1024
    character_semantic: 512
    character_emotion: 256
    character_pose: 256
```

Success criteria:
* `config.yaml` loads with new fields without validation errors
* Weights sum to 1.0

Context references:
* `config.yaml` (Lines 1-50) — Current config structure

Dependencies:
* Step 1.1 (config.py models must accept the new fields)

### Step 1.3: Update .env.example with Florence credentials

Files:
* `.env.example` — Add Azure Computer Vision credential templates

Changes:

```env
# Azure Computer Vision (Florence image embeddings)
AZURE_CV_ENDPOINT=https://your-cv-resource.cognitiveservices.azure.com
AZURE_CV_API_KEY=your-cv-api-key-here
```

Success criteria:
* `.env.example` documents all required credentials

Dependencies:
* None

## Implementation Phase 2: Client & Embedding Layer

<!-- parallelizable: true -->

### Step 2.1: Add Florence client factory to clients.py

Add an async HTTP client factory for the Azure Computer Vision REST API.

Files:
* `src/ai_search/clients.py` — Add `get_cv_client()` factory

Changes:

Add import and factory function:

```python
import httpx

from ai_search.config import load_cv_secrets

@lru_cache(maxsize=1)
def get_cv_client() -> httpx.AsyncClient:
    """Return a cached async HTTP client for Azure Computer Vision."""
    secrets = load_cv_secrets()
    return httpx.AsyncClient(
        base_url=secrets.endpoint,
        headers={"Ocp-Apim-Subscription-Key": secrets.api_key},
        timeout=30.0,
    )
```

Note: `httpx.AsyncClient` is used because Florence is a REST API, not an SDK client. The `lru_cache` caches the client for connection reuse.

Success criteria:
* `get_cv_client()` returns an `httpx.AsyncClient` configured with CV endpoint and key
* Client has the API key in the `Ocp-Apim-Subscription-Key` header

Context references:
* `src/ai_search/clients.py` (Lines 1-68) — Existing client factory pattern
* Research doc (Lines 100-120) — Florence REST API pattern

Dependencies:
* Step 1.1 (`load_cv_secrets()` must exist)

### Step 2.2: Create embeddings/image.py — image embedding module

Create a new module for direct image-to-vector embedding via Azure Computer Vision 4.0 (Florence).

Files:
* `src/ai_search/embeddings/image.py` — NEW file

Content:

```python
"""Image embedding via Azure Computer Vision 4.0 (Florence).

Provides direct image-to-vector and text-to-image-space embedding
using the Florence foundation model's multimodal embedding space.
"""

from __future__ import annotations

import structlog

from ai_search.clients import get_cv_client
from ai_search.config import load_cv_secrets

logger = structlog.get_logger(__name__)

_MODEL_VERSION = "2023-04-15"


async def embed_image(
    image_url: str | None = None,
    image_bytes: bytes | None = None,
) -> list[float]:
    """Embed an image into a 1024-dim vector via Florence vectorizeImage.

    Args:
        image_url: Public URL of the image.
        image_bytes: Raw image bytes (JPEG/PNG).

    Returns:
        1024-dimensional float vector in Florence's shared image-text space.

    Raises:
        ValueError: If neither image_url nor image_bytes is provided.
        httpx.HTTPStatusError: If the Florence API returns an error.
    """
    if not image_url and not image_bytes:
        msg = "Either image_url or image_bytes must be provided"
        raise ValueError(msg)

    client = get_cv_client()
    secrets = load_cv_secrets()
    params = {"api-version": secrets.api_version, "model-version": _MODEL_VERSION}

    if image_url:
        response = await client.post(
            "/computervision/retrieval:vectorizeImage",
            params=params,
            json={"url": image_url},
        )
    else:
        response = await client.post(
            "/computervision/retrieval:vectorizeImage",
            params=params,
            content=image_bytes,
            headers={"Content-Type": "application/octet-stream"},
        )

    response.raise_for_status()
    vector = response.json()["vector"]

    logger.info("Image embedded via Florence", dimensions=len(vector))
    return vector


async def embed_text_for_image_search(text: str) -> list[float]:
    """Embed text into Florence's image-text space for cross-modal search.

    Uses Florence's vectorizeText endpoint so the resulting vector
    is comparable (cosine similarity) with image vectors from embed_image().

    Args:
        text: Query text to embed in the image space.

    Returns:
        1024-dimensional float vector in Florence's shared image-text space.
    """
    client = get_cv_client()
    secrets = load_cv_secrets()
    params = {"api-version": secrets.api_version, "model-version": _MODEL_VERSION}

    response = await client.post(
        "/computervision/retrieval:vectorizeText",
        params=params,
        json={"text": text},
    )
    response.raise_for_status()
    vector = response.json()["vector"]

    logger.info("Text embedded via Florence (image space)", dimensions=len(vector))
    return vector
```

Success criteria:
* `embed_image(image_url=...)` returns 1024-dim vector
* `embed_image(image_bytes=...)` returns 1024-dim vector
* `embed_text_for_image_search(text)` returns 1024-dim vector in same space
* Both raise `ValueError` on invalid input
* Both raise `httpx.HTTPStatusError` on API errors

Context references:
* Research doc (Lines 100-130) — Florence REST API code examples
* `src/ai_search/embeddings/encoder.py` (Lines 1-58) — Existing embedding pattern

Dependencies:
* Step 2.1 (`get_cv_client()` and `load_cv_secrets()`)

### Step 2.3: Update embeddings/pipeline.py to include image embedding

Modify the embedding pipeline orchestrator to include the Florence image embedding call alongside the existing text embeddings.

Files:
* `src/ai_search/embeddings/pipeline.py` — Add image embedding to `generate_all_vectors()`

Changes:

1. Add import for `embed_image`:

```python
from ai_search.embeddings.image import embed_image
```

2. Add `image_url` and `image_bytes` parameters to `generate_all_vectors()`:

```python
async def generate_all_vectors(
    extraction: ImageExtraction,
    image_url: str | None = None,
    image_bytes: bytes | None = None,
) -> ImageVectors:
```

3. Add image embedding task to `asyncio.gather()`:

```python
    image_task = embed_image(image_url=image_url, image_bytes=image_bytes) if (image_url or image_bytes) else None

    tasks = [semantic_task, structural_task, style_task, character_task]
    if image_task:
        tasks.append(image_task)

    results = await asyncio.gather(*tasks)

    image_vec = results[4] if image_task else []

    return ImageVectors(
        semantic_vector=results[0],
        structural_vector=results[1],
        style_vector=results[2],
        character_vectors=results[3],
        image_vector=image_vec,
    )
```

Success criteria:
* `generate_all_vectors()` accepts optional `image_url`/`image_bytes` params
* Florence embedding runs in parallel with text embeddings via `asyncio.gather()`
* `ImageVectors.image_vector` is populated when image input is provided
* `image_vector` is empty list when no image input (backward compatible)

Context references:
* `src/ai_search/embeddings/pipeline.py` (Lines 1-31) — Current orchestrator
* Research doc (Lines 340-350) — "Florence runs in parallel with GPT-4o extraction (~0ms added latency)"

Dependencies:
* Step 2.2 (`embed_image()` function)
* Step 3.1 (`ImageVectors.image_vector` field)

## Implementation Phase 3: Index Schema & Models

<!-- parallelizable: true -->

### Step 3.1: Add image_vector to SearchDocument and ImageVectors models

Files:
* `src/ai_search/models.py` — Add `image_vector` field to `ImageVectors` and `SearchDocument`

Changes:

1. Add to `ImageVectors` (after `style_vector`, ~Line 102):

```python
class ImageVectors(BaseModel):
    """All embedding vectors for an image."""

    semantic_vector: list[float]
    structural_vector: list[float]
    style_vector: list[float]
    image_vector: list[float] = Field(default_factory=list)
    character_vectors: list[CharacterVectors] = Field(default_factory=list)
```

2. Add to `SearchDocument` (after `style_vector`, ~Line 122):

```python
    # Vectors (populated by embedding pipeline)
    semantic_vector: list[float] = Field(default_factory=list)
    structural_vector: list[float] = Field(default_factory=list)
    style_vector: list[float] = Field(default_factory=list)
    image_vector: list[float] = Field(default_factory=list)
```

Success criteria:
* `ImageVectors` has `image_vector` field defaulting to empty list
* `SearchDocument` has `image_vector` field defaulting to empty list
* Backward compatible — existing code that doesn't set `image_vector` still works

Context references:
* `src/ai_search/models.py` (Lines 93-130) — Current model structure

Dependencies:
* None — model changes are additive

### Step 3.2: Add image_vector field and scoring profile to indexing/schema.py

Add the `image_vector` HNSW field and a text-boost scoring profile to the index schema.

Files:
* `src/ai_search/indexing/schema.py` — Add vector field + scoring profile

Changes:

1. Add import for scoring profile types:

```python
from azure.search.documents.indexes.models import (
    # ... existing imports ...
    ScoringProfile,
    TextWeights,
)
```

2. Add `image_vector` field after the primary vector fields (after ~Line 109):

```python
    # Primary vector fields
    fields.extend([
        _build_vector_field("semantic_vector", dims.semantic),
        _build_vector_field("structural_vector", dims.structural),
        _build_vector_field("style_vector", dims.style),
        _build_vector_field("image_vector", dims.image),  # Florence direct embedding
    ])
```

3. Add scoring profile before the return statement (~Line 135):

```python
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
```

4. Add `scoring_profiles` to `SearchIndex` constructor:

```python
    return SearchIndex(
        name=idx.name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
        scoring_profiles=[text_boost_profile],
        default_scoring_profile="text-boost",
    )
```

Success criteria:
* Index schema includes `image_vector` field with 1024 dims and HNSW cosine
* Index has a `text-boost` scoring profile that boosts `generation_prompt` (3x) and `tags` (2x)
* Scoring profile is set as default

Context references:
* `src/ai_search/indexing/schema.py` (Lines 97-140) — Current schema definition
* Azure AI Search docs — ScoringProfile, TextWeights

Dependencies:
* Step 1.1 (`VectorDimensionsConfig.image` must exist)

## Implementation Phase 4: Retrieval Layer

<!-- parallelizable: true -->

### Step 4.1: Update retrieval/query.py to support image query vectors

Add image query embedding capability. When a text query is given, generate an image-space vector via Florence `vectorizeText` for cross-modal search.

Files:
* `src/ai_search/retrieval/query.py` — Add image query vector generation

Changes:

1. Add imports:

```python
from ai_search.embeddings.image import embed_image, embed_text_for_image_search
```

2. Add image-space text vector to `generate_query_vectors()`:

```python
async def generate_query_vectors(
    query_text: str,
    query_image_url: str | None = None,
) -> dict[str, list[float]]:
```

3. After existing embed calls, add image embedding:

```python
    # Image-space vector (for cross-modal matching)
    if query_image_url:
        # Direct image query — embed the image
        image_vec = await embed_image(image_url=query_image_url)
    else:
        # Text query — embed in image space for cross-modal search
        image_vec = await embed_text_for_image_search(query_text)

    return {
        "semantic_vector": semantic_vec,
        "structural_vector": structural_vec,
        "style_vector": style_vec,
        "image_vector": image_vec,
    }
```

4. Update `generate_query_vectors_sync()`:

```python
def generate_query_vectors_sync(
    query_text: str,
    query_image_url: str | None = None,
) -> dict[str, list[float]]:
    """Synchronous wrapper for query vector generation."""
    return asyncio.run(generate_query_vectors(query_text, query_image_url))
```

Success criteria:
* Text query generates 4 vectors: semantic, structural, style, image (cross-modal via Florence)
* Image query generates 4 vectors: semantic/structural/style (from text), image (from Florence image)
* `image_vector` key is always present in returned dict

Context references:
* `src/ai_search/retrieval/query.py` (Lines 1-78) — Current query module
* Research doc (Lines 300-310) — Query architecture diagram

Dependencies:
* Step 2.2 (`embed_image()`, `embed_text_for_image_search()`)

### Step 4.2: Update retrieval/search.py to include image_vector in hybrid search

Add `image_vector` as a fourth vector query in the hybrid search with the configured `image_weight`.

Files:
* `src/ai_search/retrieval/search.py` — Add image vector query

Changes:

1. Add `image_vector` to `SELECT_FIELDS` (needed for MMR diversity fallback):

```python
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
    "semantic_vector",   # For MMR diversity (fixes IV-014 from review)
    "image_vector",      # For MMR diversity
]
```

2. Add image vector query block (after style vector query, ~Line 80):

```python
    if "image_vector" in query_vectors:
        vector_queries.append(
            VectorizedQuery(
                vector=query_vectors["image_vector"],
                k_nearest_neighbors=retrieval.stage1_k_nearest,
                fields="image_vector",
                weight=weights.image_weight * 10,
            )
        )
```

Success criteria:
* `image_vector` is included as a weighted vector query in RRF when present
* Weight follows the same ×10 pattern as other vectors
* `SELECT_FIELDS` includes `semantic_vector` and `image_vector` (also fixes IV-014 from prior review)

Context references:
* `src/ai_search/retrieval/search.py` (Lines 14-95) — Current hybrid search
* `.copilot-tracking/reviews/2026-02-26/ai-search-pipeline-plan-review.md` — IV-014 finding

Dependencies:
* Step 1.1 (`SearchWeightsConfig.image_weight` must exist)

## Implementation Phase 5: Dependencies & Documentation

<!-- parallelizable: true -->

### Step 5.1: Verify httpx dependency in pyproject.toml

`httpx>=0.27` is already listed as a runtime dependency in `pyproject.toml`. Verify it's present and sufficient for the Florence REST calls.

Files:
* `pyproject.toml` — Verify `httpx>=0.27` is present

Success criteria:
* `httpx>=0.27` is in `dependencies` list
* No additional dependencies needed for Florence (REST API via httpx)

Dependencies:
* None

### Step 5.2: Update README.md with image embedding setup

Files:
* `README.md` — Add Florence setup instructions and S2 tier guidance

Add a section explaining:
1. Azure Computer Vision resource provisioning
2. `.env` configuration for `AZURE_CV_ENDPOINT` and `AZURE_CV_API_KEY`
3. How image embeddings work alongside text embeddings
4. That Cohere Embed v4 is the documented alternative (reference research doc)
5. S2 Standard tier recommendation: supports ~1 TB vector storage, accommodates the additional `image_vector` field (~37 GB at 10M documents)
6. Foundry-only waiver note: Florence is a separate Azure Cognitive Services resource (see DD-01 in planning log)

Success criteria:
* README documents Florence setup requirements
* README explains the dual-model architecture

Dependencies:
* None

## Implementation Phase 6: Tests

<!-- parallelizable: true -->

### Step 6.1: Add unit tests for embeddings/image.py

Files:
* `tests/test_embeddings/test_image.py` — NEW file

Tests to create:
1. `test_embed_image_url` — Mock Florence API, verify returns 1024-dim vector from URL input
2. `test_embed_image_bytes` — Mock Florence API, verify returns 1024-dim vector from bytes input
3. `test_embed_image_no_input_raises` — Verify `ValueError` when neither URL nor bytes provided
4. `test_embed_text_for_image_search` — Mock Florence `vectorizeText`, verify returns 1024-dim vector
5. `test_embed_image_api_error` — Mock 500 response, verify `httpx.HTTPStatusError` raised

Use `pytest-httpx` or manual `httpx.AsyncClient` mocking with `unittest.mock.AsyncMock`.

Mock pattern:
```python
@pytest.fixture
def mock_cv_client(monkeypatch):
    """Mock the Florence CV client."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"vector": [0.1] * 1024}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    monkeypatch.setattr("ai_search.embeddings.image.get_cv_client", lambda: mock_client)
    monkeypatch.setattr("ai_search.embeddings.image.load_cv_secrets", lambda: MagicMock(api_version="2024-02-01"))
    return mock_client
```

Success criteria:
* 5 tests covering happy path, error cases, and both input types
* All tests pass without Azure credentials

Context references:
* `tests/test_embeddings/test_encoder.py` — Existing embedding test pattern

Dependencies:
* Step 2.2 (`embeddings/image.py`)

### Step 6.2: Update existing embedding pipeline test

Files:
* `tests/test_embeddings/test_pipeline.py` — Update for image vector

Update the existing pipeline test to verify `image_vector` is populated when image URL is provided.

Success criteria:
* Pipeline test verifies `image_vector` is non-empty when image input given
* Pipeline test verifies `image_vector` is empty when no image input (backward compat)

Dependencies:
* Step 2.3 (pipeline changes)

### Step 6.3: Add unit tests for updated search and query modules

Files:
* `tests/test_retrieval/test_search.py` — Update for image vector query
* `tests/test_retrieval/test_query.py` — NEW file for query vector generation

Search test update:
* Verify `image_vector` query is included when present in `query_vectors`
* Verify `image_weight * 10` is used as the weight

Query test:
* Mock LLM + embedding + Florence clients
* Verify `generate_query_vectors()` returns dict with `image_vector` key
* Verify image-only query calls `embed_image()` instead of `embed_text_for_image_search()`

Success criteria:
* Updated search test passes with image vector
* New query test verifies cross-modal embedding generation

Dependencies:
* Steps 4.1, 4.2

## Implementation Phase 7: Validation

<!-- parallelizable: false -->

### Step 7.1: Run full project validation

Execute all validation commands for the project:
* `uv run ruff check src/ tests/`
* `uv run mypy src/`
* `uv run pytest tests/ -m "not integration"`

### Step 7.2: Fix minor validation issues

Iterate on lint errors, build warnings, and test failures. Apply fixes directly when corrections are straightforward and isolated.

### Step 7.3: Report blocking issues

When validation failures require changes beyond minor fixes:
* Document the issues and affected files.
* Provide the user with next steps.
* Recommend additional research and planning rather than inline fixes.

## Dependencies

* `httpx>=0.27` (already in deps)
* Azure Computer Vision 4.0 resource (user must provision)

## Success Criteria

* All unit tests pass (existing + new)
* Ruff and mypy pass clean
* Image embedding module works with mocked Florence API
* Config properly separates Florence credentials into `.env`
