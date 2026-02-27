<!-- markdownlint-disable-file -->
# Implementation Details: Fix Image Embedding Client

## Context Reference

Sources: `.copilot-tracking/research/2026-02-26/embedding-vector-correctness-research.md`, `src/ai_search/clients.py`, `src/ai_search/embeddings/image.py`, `tests/test_embeddings/test_image.py`

## Implementation Phase 1: Fix Image Embedding Client

<!-- parallelizable: true (steps 1.1→1.2 are sequential; step 1.3 is independent) -->

### Step 1.1: Add `ImageEmbeddingsClient` import and factory to `clients.py`

Add `ImageEmbeddingsClient` to the SDK import and create a new factory function following the same pattern as `get_foundry_embed_client()`.

Files:
* `src/ai_search/clients.py` - Add import on line 9, add factory function after line 108

Code changes:

**Line 9** — Update import to include `ImageEmbeddingsClient`:
```python
# Before (line 9):
from azure.ai.inference.aio import EmbeddingsClient

# After:
from azure.ai.inference.aio import EmbeddingsClient, ImageEmbeddingsClient
```

**After line 108** — Add new factory function:
```python
@lru_cache(maxsize=1)
def get_foundry_image_embed_client() -> ImageEmbeddingsClient:
    """Return a cached Azure AI Inference ImageEmbeddingsClient.

    Uses the Azure AI Inference SDK with ``DefaultAzureCredential`` for
    Entra ID authentication. Routes to ``POST /images/embeddings`` for
    visual image embedding (as opposed to text tokenization).
    """
    secrets = load_foundry_secrets()
    if not secrets.embed_endpoint:
        msg = (
            "AZURE_FOUNDRY_EMBED_ENDPOINT is not configured. "
            "Set it to the Foundry models endpoint, e.g. "
            "https://<resource>.services.ai.azure.com/models"
        )
        raise ValueError(msg)
    return ImageEmbeddingsClient(
        endpoint=secrets.embed_endpoint,
        credential=_get_credential(),
        credential_scopes=[_AZURE_COGNITIVE_SCOPE],
    )
```

Discrepancy references:
* Addresses DR-01 (endpoint routing uncertainty) — uses same base endpoint; SDK handles route differentiation

Success criteria:
* `get_foundry_image_embed_client()` returns an `ImageEmbeddingsClient` instance
* Function is importable from `ai_search.clients`
* Existing `get_foundry_embed_client()` remains unchanged

Context references:
* `src/ai_search/clients.py` (Lines 89-108) - Existing `get_foundry_embed_client()` as pattern reference
* Research doc (Lines 125-140) - Correct factory implementation

Dependencies:
* `azure-ai-inference>=1.0.0b9` installed (confirmed)

### Step 1.2: Update `_embed_image_foundry()` in `image.py` to use image client with `ImageEmbeddingInput`

Switch from `EmbeddingsClient` (text) to `ImageEmbeddingsClient` (image) and wrap the data URI in `ImageEmbeddingInput`.

Files:
* `src/ai_search/embeddings/image.py` - Update import on line 21, add `ImageEmbeddingInput` import, modify `_embed_image_foundry()` at lines 176-195

Code changes:

**Line 21** — Update client import:
```python
# Before (line 21):
from ai_search.clients import get_cv_client, get_foundry_embed_client

# After:
from ai_search.clients import get_cv_client, get_foundry_embed_client, get_foundry_image_embed_client
```

**Add new import** (after line 21 or with other SDK imports):
```python
from azure.ai.inference.models import ImageEmbeddingInput
```

**Lines 171-195** — Rewrite `_embed_image_foundry()`:
```python
async def _embed_image_foundry(
    model: str,
    dimensions: int,
    image_url: str | None = None,
    image_bytes: bytes | None = None,
) -> list[float]:
    """Embed image via Azure AI Inference ImageEmbeddingsClient.

    Uses the ``/images/embeddings`` route with ``ImageEmbeddingInput`` to
    ensure the model processes the image visually rather than tokenizing
    the base64 string as text.
    """
    data_uri = await _image_to_data_uri(image_url=image_url, image_bytes=image_bytes)
    client = get_foundry_image_embed_client()
    response = await client.embed(
        input=[ImageEmbeddingInput(image=data_uri)],
        model=model,
        dimensions=dimensions,
    )
    vector = list(response.data[0].embedding)
    return _validate_vector(
        vector, dimensions, f"Foundry {model} image embedding", [],
    )
```

Key differences from current code:
1. Client: `get_foundry_image_embed_client()` instead of `get_foundry_embed_client()`
2. Input: `[ImageEmbeddingInput(image=data_uri)]` instead of `[data_uri]`
3. Route: SDK sends to `POST /images/embeddings` instead of `POST /embeddings`
4. Docstring: Updated to reflect the correct API route

Discrepancy references:
* Core fix for the root cause bug identified in research

Success criteria:
* `_embed_image_foundry()` calls `ImageEmbeddingsClient.embed()` with `ImageEmbeddingInput`
* Return type remains `list[float]` with correct dimensions
* Validation logic unchanged

Context references:
* `src/ai_search/embeddings/image.py` (Lines 176-195) - Current broken implementation
* Research doc (Lines 145-160) - Correct embed call syntax

Dependencies:
* Step 1.1 completion (factory function must exist)

### Step 1.3: (Optional) Add `input_type` parameter to `_embed_text_foundry()`

Cohere Embed v4 supports `input_type` to optimize embeddings for retrieval. Setting `input_type="query"` for search queries and `input_type="document"` for stored text improves retrieval quality.

Files:
* `src/ai_search/embeddings/image.py` - Modify `_embed_text_foundry()` at lines 198-210
* `src/ai_search/embeddings/text.py` - If text embeddings also use Foundry, add `input_type="document"` there

Code changes for `_embed_text_foundry()`:
```python
async def _embed_text_foundry(model: str, text: str, dimensions: int) -> list[float]:
    """Embed text via Azure AI Inference EmbeddingsClient."""
    client = get_foundry_embed_client()
    response = await client.embed(
        input=[text],
        model=model,
        dimensions=dimensions,
        input_type="query",  # Optimize for search query retrieval
    )
    vector = list(response.data[0].embedding)
    return _validate_vector(
        vector, dimensions, f"Foundry {model} text embedding", [],
    )
```

Discrepancy references:
* DR-02 (input_type optimization) — enhancement, not bug fix

Success criteria:
* `input_type` parameter passed to `embed()` call
* No functional regression (text embeddings still produce correct vectors)

Context references:
* Research doc (Lines 170-180) - `input_type` documentation

Dependencies:
* None (independent of image client fix)

## Implementation Phase 2: Update Tests

<!-- parallelizable: true (steps within this phase are sequential: 2.1 fixture first, then 2.2/2.3) -->

### Step 2.1: Update `mock_foundry_backend` fixture to mock `ImageEmbeddingsClient`

The fixture at line 49 of `test_image.py` currently mocks `get_foundry_embed_client`. It must mock both the text client AND the new image client, since image embedding now uses a separate client.

Files:
* `tests/test_embeddings/test_image.py` - Modify `mock_foundry_backend` fixture at lines 49-72

Code changes:
```python
@pytest.fixture()
def mock_foundry_backend(monkeypatch: pytest.MonkeyPatch) -> dict[str, AsyncMock]:
    """Mock Foundry backend: config, image download, and embedding clients."""
    config = MagicMock()
    config.models.image_embedding_model = "embed-v-4-0"
    config.index.vector_dimensions.image = 1024
    monkeypatch.setattr("ai_search.embeddings.image.load_config", lambda: config)

    # Mock _image_to_data_uri so we don't download anything
    monkeypatch.setattr(
        "ai_search.embeddings.image._image_to_data_uri",
        AsyncMock(return_value="data:image/jpeg;base64,AAAA"),
    )

    # Mock Azure AI Inference ImageEmbeddingsClient (image route)
    embedding_obj = MagicMock()
    embedding_obj.embedding = [0.2] * 1024
    embed_response = MagicMock()
    embed_response.data = [embedding_obj]
    image_embed_client = AsyncMock()
    image_embed_client.embed = AsyncMock(return_value=embed_response)
    monkeypatch.setattr(
        "ai_search.embeddings.image.get_foundry_image_embed_client",
        lambda: image_embed_client,
    )

    # Mock Azure AI Inference EmbeddingsClient (text route, for text queries)
    text_embed_client = AsyncMock()
    text_embed_client.embed = AsyncMock(return_value=embed_response)
    monkeypatch.setattr(
        "ai_search.embeddings.image.get_foundry_embed_client",
        lambda: text_embed_client,
    )

    return {
        "image_embed_client": image_embed_client,
        "text_embed_client": text_embed_client,
    }
```

Success criteria:
* Fixture mocks both `get_foundry_image_embed_client` (for image tests) and `get_foundry_embed_client` (for text tests)
* Return dict uses descriptive keys: `image_embed_client`, `text_embed_client`

Context references:
* `tests/test_embeddings/test_image.py` (Lines 49-72) - Current fixture

Dependencies:
* Step 1.1 and 1.2 must be implemented (new import paths exist)

### Step 2.2: Update Foundry image test assertions for `ImageEmbeddingInput`

Tests in `TestFoundryEmbedImage` (lines 260-285) assert `call_kwargs["input"] == ["data:image/jpeg;base64,AAAA"]`. After the fix, the input will be `[ImageEmbeddingInput(image="data:image/jpeg;base64,AAAA")]`.

Files:
* `tests/test_embeddings/test_image.py` - Modify tests in `TestFoundryEmbedImage` class (lines 231-256)

Code changes for `test_embed_image_url` (line 266):
```python
@pytest.mark.asyncio()
async def test_embed_image_url(self, mock_foundry_backend: dict[str, AsyncMock]) -> None:
    """Should return 1024-dim vector from image URL via Foundry."""
    result = await embed_image(image_url="https://example.com/test.jpg")

    assert len(result) == 1024
    image_embed_client = mock_foundry_backend["image_embed_client"]
    image_embed_client.embed.assert_called_once()
    call_kwargs = image_embed_client.embed.call_args.kwargs
    assert call_kwargs["model"] == "embed-v-4-0"
    assert call_kwargs["dimensions"] == 1024
    # Input should be ImageEmbeddingInput, not a plain string
    inputs = call_kwargs["input"]
    assert len(inputs) == 1
    assert inputs[0].image == "data:image/jpeg;base64,AAAA"
```

Code changes for `test_embed_image_bytes` (line 277):
```python
@pytest.mark.asyncio()
async def test_embed_image_bytes(self, mock_foundry_backend: dict[str, AsyncMock]) -> None:
    """Should convert bytes to data URI and embed via Foundry."""
    result = await embed_image(image_bytes=b"fake-png-data")

    assert len(result) == 1024
    image_embed_client = mock_foundry_backend["image_embed_client"]
    call_kwargs = image_embed_client.embed.call_args.kwargs
    inputs = call_kwargs["input"]
    assert len(inputs) == 1
    assert inputs[0].image == "data:image/jpeg;base64,AAAA"
```

Success criteria:
* Tests assert against `image_embed_client` (not `embed_client`)
* Input assertions check `ImageEmbeddingInput.image` attribute, not raw string
* Both URL and bytes paths tested

Context references:
* `tests/test_embeddings/test_image.py` (Lines 260-285) - Current image tests

Dependencies:
* Step 2.1 completion (fixture must return `image_embed_client`)

### Step 2.3: Update Foundry validation test to mock the image client

The `TestFoundryValidation.test_invalid_image_response_raises` test (line 303) mocks `get_foundry_embed_client` for an image embedding error scenario. It must mock `get_foundry_image_embed_client` instead.

Files:
* `tests/test_embeddings/test_image.py` - Modify test at lines 303-331

Code changes:
```python
@pytest.mark.asyncio()
async def test_invalid_image_response_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
    """Should raise ValueError when Foundry image response has wrong dims."""
    config = MagicMock()
    config.models.image_embedding_model = "embed-v-4-0"
    config.index.vector_dimensions.image = 1024
    monkeypatch.setattr("ai_search.embeddings.image.load_config", lambda: config)

    # Mock _image_to_data_uri
    monkeypatch.setattr(
        "ai_search.embeddings.image._image_to_data_uri",
        AsyncMock(return_value="data:image/jpeg;base64,AAAA"),
    )

    # Mock ImageEmbeddingsClient returning wrong dimensions
    embedding_obj = MagicMock()
    embedding_obj.embedding = [0.1] * 512
    embed_response = MagicMock()
    embed_response.data = [embedding_obj]
    embed_client = AsyncMock()
    embed_client.embed = AsyncMock(return_value=embed_response)
    monkeypatch.setattr(
        "ai_search.embeddings.image.get_foundry_image_embed_client",
        lambda: embed_client,
    )

    with pytest.raises(ValueError, match="expected 1024-dim vector, got 512"):
        await embed_image(image_url="https://example.com/test.jpg")
```

Also update `TestFoundryEmbedText` (line 263) to use `text_embed_client`:
```python
@pytest.mark.asyncio()
async def test_embed_text(self, mock_foundry_backend: dict[str, AsyncMock]) -> None:
    """Should return 1024-dim vector from text via Azure AI Inference SDK."""
    result = await embed_text_for_image_search("woman in red dress")

    assert len(result) == 1024
    text_embed_client = mock_foundry_backend["text_embed_client"]
    text_embed_client.embed.assert_called_once()
    call_kwargs = text_embed_client.embed.call_args.kwargs
    assert call_kwargs["model"] == "embed-v-4-0"
    assert call_kwargs["dimensions"] == 1024
```

Success criteria:
* Validation test mocks `get_foundry_image_embed_client` instead of `get_foundry_embed_client`
* Text test uses `text_embed_client` key from updated fixture
* ValueError still raised for wrong dimensions

Context references:
* `tests/test_embeddings/test_image.py` (Lines 287-300) - Text test
* `tests/test_embeddings/test_image.py` (Lines 303-331) - Validation test

Dependencies:
* Step 2.1 completion (fixture must provide both clients)

## Implementation Phase 3: Re-ingest Documents

<!-- parallelizable: false -->

### Step 3.1: Delete existing index or clear image_vector values

The existing `image_vector` values in all 10 documents are invalid (text-tokenized base64 embeddings). Either delete the entire index and recreate it, or update the ingestion script with a `--force` flag.

Files:
* `scripts/ingest_samples.py` - Add `--force` flag to bypass the "already indexed" skip logic (lines 152-160)

Code changes — add `--force` argument:
```python
parser.add_argument(
    "--force",
    action="store_true",
    help="Re-index all images, even those already in the index",
)
```

And modify the skip logic in `run()`:
```python
async def run(dry_run: bool = False, force: bool = False) -> None:
    # ... existing code ...
    if not force:
        # Check which docs are already indexed to skip them
        # ... existing skip logic ...
    else:
        already_indexed = set()
```

Alternative: Simpler approach is to delete and recreate the index using the existing `scripts/create_index.py`, then re-run `ingest_samples.py` without changes.

Success criteria:
* All 10 documents are re-indexed with new image vectors
* Text-based vectors (semantic, structural, style) remain correct

Context references:
* `scripts/ingest_samples.py` (Lines 140-165) - Skip logic

Dependencies:
* Phase 1 must be complete (code fix must be in place before re-indexing)

### Step 3.2: Run ingestion pipeline with force flag

Execute the ingestion pipeline with all 10 sample images.

Commands:
```bash
source .venv/bin/activate
SSL_CERT_FILE=/private/etc/ssl/cert.pem python scripts/ingest_samples.py --force
```

Or if using the delete-and-recreate approach:
```bash
source .venv/bin/activate
SSL_CERT_FILE=/private/etc/ssl/cert.pem python scripts/create_index.py
SSL_CERT_FILE=/private/etc/ssl/cert.pem python scripts/ingest_samples.py
```

Note: The S0 tier rate limits apply. The script already has `INTER_IMAGE_DELAY_S = 10` and `RETRY_BACKOFF_S = 30` with `MAX_RETRIES = 5`.

Success criteria:
* All 10 documents indexed successfully
* Embedding dimensions validated: semantic=3072, structural=1024, style=512, image=1024
* No rate limit failures after retries

Dependencies:
* Step 3.1 completion

### Step 3.3: Verify new image vectors produce differentiated scores

Run the existing test scripts to validate that image search now differentiates content.

Commands:
```bash
source .venv/bin/activate
SSL_CERT_FILE=/private/etc/ssl/cert.pem python scripts/test_image_search.py
SSL_CERT_FILE=/private/etc/ssl/cert.pem python scripts/analyze_scores.py
SSL_CERT_FILE=/private/etc/ssl/cert.pem python scripts/test_relevance_tiers.py
```

Expected results:
* Ocean image → sample-003: score ~1.0 (exact match)
* Ocean image → unrelated: scores 0.3-0.7 (not 0.95+)
* Cat image → all: scores 0.3-0.6 (not in index, irrelevant)
* Mountain → mountain/landscape images: scores 0.7-0.9 (similar)
* Score spread > 0.3 (currently < 0.05)

Success criteria:
* Unrelated images score below 0.7
* Exact matches score above 0.9
* Score spread allows meaningful threshold setting
* Relevance tiers correctly classify: exact=HIGH, similar=MEDIUM, irrelevant=LOW

Dependencies:
* Step 3.2 completion (documents re-indexed)

## Implementation Phase 4: Validation

<!-- parallelizable: false -->

### Step 4.1: Run full project validation

Execute all validation commands for the project:
* `source .venv/bin/activate && python -m pytest tests/ -v` — all 50+ tests must pass
* `source .venv/bin/activate && python -m pytest tests/test_embeddings/test_image.py -v` — image embedding tests specifically

### Step 4.2: Fix minor validation issues

Iterate on test failures and lint errors. Apply fixes directly when corrections are straightforward and isolated.

### Step 4.3: Report blocking issues

When validation failures require changes beyond minor fixes:
* Document the issues and affected files
* Provide the user with next steps
* Recommend additional research rather than inline fixes
* If `ImageEmbeddingsClient` endpoint routing differs from `EmbeddingsClient`, investigate and document

## Dependencies

* `azure-ai-inference>=1.0.0b9` (already installed)

## Success Criteria

* All unit tests pass with updated mocks for `ImageEmbeddingsClient`
* Image embeddings generated via `/images/embeddings` route produce visually meaningful vectors
* Cosine similarity between unrelated images drops from 0.95+ to 0.3-0.7
* Relevance scoring tiers work correctly for image search
