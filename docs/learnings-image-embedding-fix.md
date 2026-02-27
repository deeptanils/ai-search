# Learnings: Image Embedding Vector Correctness

Date: 2026-02-26

## Problem Statement

Image search returned uniformly high cosine similarity scores (0.95+) for all queries — including completely unrelated images. A cat photo scored 0.967 against an ocean image. No meaningful threshold could separate relevant from irrelevant results.

---

## Root Cause

### Wrong SDK Client for Image Embeddings

The Azure AI Inference SDK provides **two separate embedding clients**:

| Client | Route | Input Type | Purpose |
|--------|-------|------------|---------|
| `EmbeddingsClient` | `POST /embeddings` | `List[str]` | **Text** embeddings |
| `ImageEmbeddingsClient` | `POST /images/embeddings` | `List[ImageEmbeddingInput]` | **Image** embeddings |

The codebase used `EmbeddingsClient` for images, passing base64 data URIs as plain strings. The model tokenized the base64 characters as text instead of processing the image visually.

### Why All Scores Were High

All base64 data URI strings share structural features:
- Same prefix: `data:image/jpeg;base64,`
- Same character set: `A-Za-z0-9+/=`
- Similar length after resizing to 512×512

When tokenized as text, the resulting embeddings were dominated by shared base64 character distribution — not visual content. This produced nearly identical vectors regardless of actual image content.

---

## Fix Applied

### Code Changes

1. **`src/ai_search/clients.py`** — Added `ImageEmbeddingsClient` import and `get_foundry_image_embed_client()` factory
2. **`src/ai_search/embeddings/image.py`** — Switched `_embed_image_foundry()` to use:
   - `get_foundry_image_embed_client()` instead of `get_foundry_embed_client()`
   - `ImageEmbeddingInput(image=data_uri)` wrapper instead of raw string
   - Added `from azure.ai.inference.models import ImageEmbeddingInput`
3. **`tests/test_embeddings/test_image.py`** — Updated fixture and assertions to mock `ImageEmbeddingsClient` separately from `EmbeddingsClient`
4. **`scripts/ingest_samples.py`** — Added `--force` flag for re-indexing already indexed documents

### Before vs After Fix

| Query | Before | After |
|-------|--------|-------|
| Ocean (exact match in index) | Score 1.0, gap to #2: 0.019 | Score 1.0, gap to #2: **0.32** |
| Cat (not in index, irrelevant) | Score **0.967** | Score **0.54** |
| Mountain (similar content) | Score 0.95+ | Score **0.61** (correct match) |
| Score spread across results | 0.02–0.05 | **0.12–0.47** |

---

## Key Learnings

### 1. Azure AI Inference SDK Has Separate Clients for Text and Images

The SDK explicitly separates text and image embedding into different classes with different API routes. There is no unified multimodal client. Always use:
- `EmbeddingsClient` → text only
- `ImageEmbeddingsClient` + `ImageEmbeddingInput` → images

### 2. Base64 Strings Are Not Images to a Text Embedding Model

Passing a base64 data URI string to a text embedding endpoint does not embed the image. The model tokenizes the base64 characters. Since all base64 strings share the same character set and structure, they produce nearly identical embeddings — which is why cosine similarity was 0.95+ for all pairs.

### 3. Uniformly High Scores Are a Red Flag

When all cosine similarity scores cluster above 0.9 regardless of query content, the embedding pipeline is likely broken. Correct multimodal embeddings produce:
- Exact match: ~1.0
- Similar content: 0.6–0.8
- Unrelated content: 0.3–0.6

### 4. Absolute Thresholds Need Correct Embeddings First

Relative metrics (z-score, gap ratio, spread) can partially compensate for compressed score distributions, but they cannot replace correct embeddings. Fix the embedding pipeline before tuning thresholds.

### 5. Small Corpus Limitations on Relevance Scoring

With only 10 documents, relative relevance metrics can produce false positives (MEDIUM confidence for irrelevant queries) because the score distributions are too compressed. Relative scoring works best with 50+ documents where score variance is meaningful.

### 6. Same Endpoint, Different Routes

`ImageEmbeddingsClient` uses the same base Foundry endpoint as `EmbeddingsClient` — the SDK handles route differentiation internally (`/embeddings` vs `/images/embeddings`). No endpoint configuration change is needed.

### 7. Re-Ingestion Required After Embedding Fix

All existing `image_vector` values generated with the text client are invalid and must be regenerated. The `--force` flag on ingestion scripts is useful for this. Text-based vectors (semantic, structural, style) remain correct since they correctly use `EmbeddingsClient` with actual text.

### 8. Image Resizing Still Matters

Even with the correct client, images should be resized before embedding (512×512, JPEG q80) to reduce payload size and stay within S0-tier rate limits. The `ImageEmbeddingsClient` benefits from the same optimization.

---

## Investigation Approach That Worked

1. **Tested search end-to-end** — Verified text search worked, then tried image search
2. **Noticed anomaly** — All image scores 0.95+ regardless of content
3. **Measured raw cosine similarity** — Confirmed the embeddings themselves were nearly identical
4. **Built relative scoring** — Attempted z-score/gap/spread metrics as workaround
5. **Questioned the embeddings** — Asked "are the vectors correctly created?"
6. **Traced the code path** — Found `EmbeddingsClient` used instead of `ImageEmbeddingsClient`
7. **Checked SDK surface** — Confirmed `ImageEmbeddingsClient` and `ImageEmbeddingInput` exist but were never used
8. **Applied fix and re-indexed** — Scores immediately differentiated

---

## Follow-Up Items

- Re-tune relevance thresholds in `relevance.py` using empirical data from corrected embeddings
- Add `input_type="query"` / `"document"` to Cohere Embed v4 text calls for retrieval optimization
- Test with 50–100+ images to validate threshold scaling
- Add integration test that validates image embedding dimensions and score differentiation against the live endpoint

---

## Other Issues Encountered During Development

### 9. Image Resizing for Base64 Payload Reduction

**Problem**: Raw images (1024×1024+ from Unsplash) produced massive base64 strings when encoded for the embedding API. This caused slow requests and quickly hit S0-tier rate limits on embed-v-4-0.

**Solution**: Added Pillow-based image resizing before base64 encoding — downscale to 512×512 max using `Image.LANCZOS`, convert to RGB, re-encode as JPEG quality 80. This achieved a **77% reduction** in base64 payload size.

**Code**: `_resize_image_bytes()` in `src/ai_search/embeddings/image.py`

```python
def _resize_image_bytes(raw: bytes, max_size: int = 512) -> bytes:
    img = Image.open(io.BytesIO(raw))
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()
```

**Lesson**: Always resize images before embedding. The visual quality loss at 512×512 is negligible for embedding purposes, but the payload savings are significant.

---

### 10. SSL Certificate Failures in Python venv

**Problem**: Python scripts running inside a venv failed with SSL certificate verification errors when calling Azure APIs. The venv's bundled `certifi` certificates were corrupt or outdated.

**Solution**: Force the system SSL certificate bundle by setting `SSL_CERT_FILE=/private/etc/ssl/cert.pem` before any network calls. Added this as an environment guard at the top of scripts:

```python
if not os.environ.get("SSL_CERT_FILE"):
    _sys_cert = "/private/etc/ssl/cert.pem"
    if os.path.exists(_sys_cert):
        os.environ["SSL_CERT_FILE"] = _sys_cert
```

**Lesson**: On macOS, venv certifi can diverge from system certificates. Always set `SSL_CERT_FILE` to the system cert bundle when working with Azure services in a venv.

---

### 11. `ModuleNotFoundError` with `uv run` vs `source .venv/bin/activate`

**Problem**: Running `uv run python scripts/ingest_samples.py` failed with `ModuleNotFoundError: No module named 'ai_search'`. The `uv run` command did not resolve the editable install properly.

**Solution**: Use `source .venv/bin/activate` then `uv pip install -e .` to install the package in editable mode, then run scripts directly with `python scripts/...`. Avoid `uv run` for scripts that import the local package.

**Lesson**: `uv run` spawns a subprocess that may not inherit editable installs. Activate the venv and install `-e .` explicitly.

---

### 12. `SearchableField` Silently Ignoring `type` Parameter for Collection Fields

**Problem**: Used `SearchableField(name="tags", type=SearchFieldDataType.Collection(SearchFieldDataType.String), filterable=True)` — but `SearchableField` silently ignored the `type` parameter and defaulted to `Edm.String` instead of `Collection(Edm.String)`. Tags were stored as a single concatenated string instead of an array.

**Solution**: Switched to `SearchField` (the base class) which correctly accepts and applies the `type` parameter:

```python
SearchField(
    name="tags",
    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
    searchable=True,
    filterable=True,
    facetable=True,
)
```

**Lesson**: Azure AI Search SDK's `SearchableField` and `SimpleField` are convenience wrappers that may silently ignore certain parameters. For complex field types like `Collection(...)`, use `SearchField` directly and set `searchable`, `filterable`, etc. explicitly.

---

### 13. AsyncOpenAI to Azure AI Inference SDK Migration

**Problem**: Initially used `AsyncOpenAI` client with the embeddings API. This required API key auth and lacked proper support for Cohere Embed v4's multimodal capabilities (image embeddings, input types).

**Solution**: Migrated to `azure.ai.inference.aio.EmbeddingsClient` (and later `ImageEmbeddingsClient`) which provides:
- Entra ID authentication via `DefaultAzureCredential`
- Proper model routing through Azure AI Foundry
- Separate clients for text vs image embedding
- `ImageEmbeddingInput` model for structured image input

**Lesson**: For Azure AI Foundry deployments, prefer the `azure-ai-inference` SDK over the OpenAI SDK. It provides first-class support for Azure auth, model routing, and multimodal inputs.

---

### 14. Entra ID Auth Scope for Azure AI Inference

**Problem**: Initial Entra ID auth attempts failed because the credential scope was not set correctly. The `DefaultAzureCredential` needs to know which Azure resource scope to request tokens for.

**Solution**: Set `credential_scopes=["https://cognitiveservices.azure.com/.default"]` when constructing the client:

```python
EmbeddingsClient(
    endpoint=secrets.embed_endpoint,
    credential=DefaultAzureCredential(),
    credential_scopes=["https://cognitiveservices.azure.com/.default"],
)
```

**Lesson**: Azure AI Foundry models use the Cognitive Services scope (`https://cognitiveservices.azure.com/.default`). This must be explicitly provided — it is not auto-detected.

---

### 15. Missing `aiohttp` Dependency for Azure AI Inference SDK

**Problem**: After installing `azure-ai-inference`, importing the async client raised `ModuleNotFoundError: No module named 'aiohttp'`. The SDK uses `aiohttp` for async HTTP but does not list it as a hard dependency.

**Solution**: Added `aiohttp` to the project dependencies and installed it:

```bash
uv pip install aiohttp
```

**Lesson**: `azure-ai-inference` async clients require `aiohttp` but may not install it automatically. Always install it alongside the SDK.

---

### 16. S0-Tier Rate Limiting on embed-v-4-0

**Problem**: Batch ingestion of 10 images hit HTTP 429 (Too Many Requests) from the embed-v-4-0 model on the S0 pricing tier. Requests were being sent too quickly in sequence.

**Solution**: Implemented three-layer rate limit handling:
1. **Inter-image delay**: 10 seconds between each image processing
2. **Retry with backoff**: On 429, wait `30s × attempt_number` before retrying (up to 5 retries)
3. **Image resizing**: Smaller payloads reduce token consumption per request

```python
INTER_IMAGE_DELAY_S = 10
MAX_RETRIES = 5
RETRY_BACKOFF_S = 30
```

**Lesson**: S0-tier Azure AI models have strict rate limits. Always implement inter-request delays and exponential backoff for batch operations. Reducing payload size (image resizing) also helps stay within limits.

---

### 17. Async Image Downloads Failing with SSL Errors

**Problem**: Using `httpx.AsyncClient` to download images inside an async function caused intermittent SSL errors, likely due to event loop and certificate context conflicts.

**Solution**: Pre-download all images synchronously with `httpx.Client` before entering the async embedding pipeline:

```python
def download_images(inputs: list[ImageInput]) -> dict[str, bytes]:
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for inp in inputs:
            resp = client.get(inp.image_url)
            downloaded[inp.image_id] = resp.content
    return downloaded
```

Then pass `image_bytes` to the async pipeline instead of URLs.

**Lesson**: When mixing sync and async code with network calls, prefer downloading data synchronously first, then pass raw bytes to async functions. This avoids SSL context issues in nested event loops.
