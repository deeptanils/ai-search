<!-- markdownlint-disable-file -->
# Task Research: Embedding Vector Correctness for Image Search

The image embedding pipeline uses embed-v-4-0 (Cohere Embed v4) via the Azure AI Inference SDK. All image search queries return uniformly high cosine similarity scores (0.95+), even for completely unrelated images (cat vs ocean). This research investigates whether the embedding vectors are correctly created.

## Task Implementation Requests

* Determine if the current image embedding code produces correct image-aware vectors
* Identify the root cause of uniformly high cosine similarity scores across unrelated images
* Provide the correct implementation for image embeddings via Azure AI Inference SDK

## Scope and Success Criteria

* Scope: `_embed_image_foundry()` in `image.py`, `get_foundry_embed_client()` in `clients.py`, and the Azure AI Inference SDK's embedding API surface
* Assumptions: embed-v-4-0 (Cohere Embed v4) supports multimodal image+text embeddings in a shared vector space
* Success Criteria:
  * Root cause of uniformly high cosine scores identified with evidence
  * Correct API call format documented with SDK class references
  * Implementation fix validated

## Outline

1. Current implementation analysis
2. Azure AI Inference SDK API surface
3. Root cause identification
4. Correct implementation
5. Impact assessment

## Research Executed

### File Analysis

* `src/ai_search/embeddings/image.py` (lines 170-210)
  * `_embed_image_foundry()` uses `get_foundry_embed_client()` which returns `EmbeddingsClient` (text embedding client)
  * Passes base64 data URI as a plain string: `input=[data_uri]`
  * No `input_type` parameter set
  * No `ImageEmbeddingInput` wrapper used

* `src/ai_search/clients.py` (lines 85-95)
  * `get_foundry_embed_client()` imports and returns `EmbeddingsClient` from `azure.ai.inference.aio`
  * Only text embedding client is instantiated; no `ImageEmbeddingsClient` exists in the codebase

* `.venv/lib/python3.12/site-packages/azure/ai/inference/aio/__init__.py`
  * Exports both `EmbeddingsClient` and `ImageEmbeddingsClient`
  * These are two distinct clients hitting different API routes

* `.venv/lib/python3.12/site-packages/azure/ai/inference/aio/_patch.py` (lines 928-940, 1176-1190)
  * `EmbeddingsClient.embed()`: `input: List[str]` → `POST /embeddings`
  * `ImageEmbeddingsClient.embed()`: `input: List[ImageEmbeddingInput]` → `POST /images/embeddings`

* `.venv/lib/python3.12/site-packages/azure/ai/inference/models/_models.py` (lines 1028-1065)
  * `ImageEmbeddingInput`: model with `image: str` (required, data URI) and `text: str` (optional)
  * Has `load()` classmethod for file-based input

### Code Search Results

* `ImageEmbeddingsClient` in workspace code: zero occurrences (never used)
* `ImageEmbeddingInput` in workspace code: zero occurrences (never used)
* `input_type` in workspace code: zero occurrences (never set in any embed call)

## Key Discoveries

### The Bug: Wrong Client, Wrong API Route

The Azure AI Inference SDK provides two separate embedding clients:

| Client | Route | `input` type | Purpose |
|---|---|---|---|
| `EmbeddingsClient` | `POST /embeddings` | `List[str]` | Text embeddings |
| `ImageEmbeddingsClient` | `POST /images/embeddings` | `List[ImageEmbeddingInput]` | Image embeddings |

The current code uses `EmbeddingsClient` for images, passing the base64 data URI as a plain string. The model receives this at the `/embeddings` route and treats it as text to tokenize, not as image data to process visually.

### Why Scores Are All High (0.95+)

All base64 data URI strings share structural features:
* Same prefix: `data:image/jpeg;base64,`
* Same character set: `A-Za-z0-9+/=`
* Similar length and token distribution after resizing to 512×512

When the text embedding model tokenizes these strings, the resulting vectors are dominated by the shared base64 character distribution rather than visual content. This produces nearly identical embeddings regardless of actual image content.

### Evidence: Test Output Comparison

Ocean image (sample-003, exact match in index):
* Image-only search: score = 1.000, gap to #2 = 0.019 (score = 0.981)
* All other images: 0.965-0.981 (tight cluster)

Cat image (not in index, completely unrelated):
* Best score: 0.967 against sample-003 (ocean!)
* All images: 0.943-0.967 (even tighter cluster)

A cat should not score 0.967 against an ocean image. With correct image embeddings, unrelated images should be in the 0.3-0.6 range.

### Correct API Usage

The fix requires three changes:

1. Import `ImageEmbeddingsClient` in `clients.py`
2. Add `get_foundry_image_embed_client()` factory function
3. Update `_embed_image_foundry()` to use the image client with `ImageEmbeddingInput`

```python
# clients.py
from azure.ai.inference.aio import EmbeddingsClient, ImageEmbeddingsClient
from azure.ai.inference.models import ImageEmbeddingInput

@lru_cache(maxsize=1)
def get_foundry_image_embed_client() -> ImageEmbeddingsClient:
    secrets = load_foundry_secrets()
    if not secrets.embed_endpoint:
        raise ValueError("AZURE_FOUNDRY_EMBED_ENDPOINT is not configured")
    return ImageEmbeddingsClient(
        endpoint=secrets.embed_endpoint,
        credential=_get_credential(),
        credential_scopes=[_AZURE_COGNITIVE_SCOPE],
    )
```

```python
# image.py - _embed_image_foundry()
from azure.ai.inference.models import ImageEmbeddingInput
from ai_search.clients import get_foundry_image_embed_client

async def _embed_image_foundry(model, dimensions, image_url=None, image_bytes=None):
    data_uri = await _image_to_data_uri(image_url=image_url, image_bytes=image_bytes)
    client = get_foundry_image_embed_client()
    response = await client.embed(
        input=[ImageEmbeddingInput(image=data_uri)],
        model=model,
        dimensions=dimensions,
    )
    vector = list(response.data[0].embedding)
    return _validate_vector(vector, dimensions, f"Foundry {model} image embedding", [])
```

### Text Embedding input_type

The `_embed_text_foundry()` function for text queries could also benefit from setting `input_type`:
* `input_type="query"` for search queries (used when searching by text against image vectors)
* `input_type="document"` for stored document text

The Cohere model uses these to optimize embedding for retrieval. This is an improvement, not a bug fix.

## Technical Scenarios

### Scenario: Switch to ImageEmbeddingsClient

The only viable approach. The SDK explicitly separates text and image embedding into different clients and API routes. There is no way to make `EmbeddingsClient` process images correctly.

**Requirements:**

* `ImageEmbeddingsClient` must be available in the installed SDK version (confirmed: azure-ai-inference>=1.0.0b9)
* `ImageEmbeddingInput` must be used to wrap data URIs
* Existing text embedding via `EmbeddingsClient` remains unchanged

**Preferred Approach:**

* Add `get_foundry_image_embed_client()` factory in `clients.py` alongside existing `get_foundry_embed_client()`
* Update `_embed_image_foundry()` to use the image client
* Update tests to mock `ImageEmbeddingsClient` instead of `EmbeddingsClient`
* Re-run ingestion pipeline to regenerate image vectors

```text
Modified files:
  src/ai_search/clients.py          — add ImageEmbeddingsClient factory
  src/ai_search/embeddings/image.py — use image client + ImageEmbeddingInput
  tests/test_embeddings/test_image.py — update mocks
```

**Implementation Details:**

After code changes, all 10 documents must be re-indexed because existing `image_vector` values were generated with the text client and contain text-tokenized base64 embeddings, not visual embeddings.

The re-ingestion can use the existing `scripts/ingest_samples.py` after deleting the current index or clearing `image_vector` values. The text-based vectors (semantic, structural, style) remain correct since they use `EmbeddingsClient` with actual text, which is the right client for text.

#### Considered Alternatives

No viable alternatives exist. The `EmbeddingsClient` (text route) cannot process image data. The SDK design explicitly separates these concerns with different clients and routes.

## Potential Next Research

* Validate that `ImageEmbeddingsClient` works with the Foundry endpoint (same base URL or different?)
  * Reasoning: The endpoint might need adjustment if the image route is mounted differently
  * Reference: Azure AI Inference SDK routing logic in `_patch.py`
* Investigate `input_type` values for text embeddings in Cohere Embed v4
  * Reasoning: Using `query` vs `document` could improve text-based search quality
  * Reference: Cohere Embed v4 documentation on search optimization
