<!-- markdownlint-disable-file -->
# IV-011: Florence API Response Validation Research

Research into defensive validation for `embed_image()` and `embed_text_for_image_search()` in `src/ai_search/embeddings/image.py`.

## 1. Current Error Handling Analysis

### What `raise_for_status()` Covers

Both functions call `response.raise_for_status()` (lines 58 and 86) before accessing the response body. This covers:

- **HTTP 4xx errors**: Bad request (400), unauthorized (401), forbidden (403), not found (404), rate-limited (429)
- **HTTP 5xx errors**: Server errors (500, 502, 503, 504)

When the Florence API returns an error HTTP status, `httpx.HTTPStatusError` is raised, which propagates to callers. This is documented in the docstring for `embed_image()`.

### What Is NOT Covered

After `raise_for_status()` passes, two unguarded operations occur:

```python
vector: list[float] = response.json()["vector"]  # Line 59 / 87
```

This line has **three unprotected failure points**:

1. **`response.json()` may raise `json.JSONDecodeError`** if the response body is not valid JSON (e.g., HTML error page, empty body, truncated response).
2. **`["vector"]` may raise `KeyError`** if the JSON response does not contain a `"vector"` key (e.g., API returns `{"error": {...}}` with a 200 status, or schema changes).
3. **The vector may be empty (`[]`)** — this would pass silently as `list[float]` but produce a zero-dimension vector that corrupts the search index (HNSW expects 1024 dimensions).

### Input Validation That Exists

`embed_image()` validates that at least one of `image_url` or `image_bytes` is provided (line 36-38). `embed_text_for_image_search()` has no input validation on the `text` parameter (empty string is accepted).

---

## 2. Florence API Response Schema

### Success Response Format

Based on the Azure Computer Vision 4.0 documentation and codebase usage (research document `model-strategy-index-config-research.md`):

**`vectorizeImage` endpoint:**

```json
{
  "vector": [0.0123, -0.0456, 0.0789, ...],
  "modelVersion": "2023-04-15"
}
```

- `vector`: Array of 1024 `float` values (fixed dimension)
- `modelVersion`: String echoing the model version used

**`vectorizeText` endpoint:**

```json
{
  "vector": [0.0123, -0.0456, 0.0789, ...],
  "modelVersion": "2023-04-15"
}
```

Same schema as `vectorizeImage` — both return 1024-dimensional vectors in a shared embedding space.

### Error Response Format

Florence returns standard Azure Cognitive Services error bodies for HTTP error responses:

```json
{
  "error": {
    "code": "InvalidRequest",
    "message": "Image URL is not accessible.",
    "innererror": {
      "code": "InvalidImageUrl",
      "message": "Image URL is not accessible."
    }
  }
}
```

Common error codes:

| HTTP Status | Error Code | Scenario |
|-------------|-----------|----------|
| 400 | `InvalidRequest` | Malformed request, bad URL, unsupported format |
| 401 | `Unauthorized` | Missing or invalid API key |
| 404 | `NotFound` | Wrong endpoint path |
| 429 | `TooManyRequests` | Rate limit exceeded |
| 500 | `InternalServerError` | Transient Azure failure |

These are all caught by `raise_for_status()`.

### Edge Cases Where `vector` Key Could Be Missing or Empty

1. **200 with error body**: Rare but possible in Azure Cognitive Services if the API returns a 200 with a partial error body (e.g., image is valid but too small to produce embeddings). No documented cases for Florence specifically, but defensive coding is warranted.
2. **API version mismatch**: Using an unsupported `model-version` could produce an unexpected response schema.
3. **Proxy/gateway interference**: Corporate proxies or API management layers can inject HTML responses with 200 status codes.
4. **Response truncation**: Network issues could truncate the JSON body.

---

## 3. Failure Modes and Downstream Impact

### Failure Mode Matrix

| Failure Mode | Exception | Current Behavior | Downstream Impact |
|-------------|-----------|-------------------|-------------------|
| HTTP error (4xx/5xx) | `httpx.HTTPStatusError` | **Handled** by `raise_for_status()` | Propagates to caller |
| Non-JSON response body | `json.JSONDecodeError` | **Unhandled** — crashes with cryptic error | Pipeline aborts with confusing traceback |
| Missing `"vector"` key | `KeyError` | **Unhandled** — crashes with `KeyError: 'vector'` | Pipeline aborts; unclear error source |
| Empty vector `[]` | None (silent) | **Unhandled** — returns `[]` | **Silent data corruption**: empty vector stored in index, HNSW field dimension mismatch, search quality degraded |
| Wrong dimension count | None (silent) | **Unhandled** — returns truncated/oversized vector | **Silent data corruption**: HNSW index may reject or misindex the document; Azure Search may return 400 at upload time |
| Non-numeric values in vector | None (silent) | **Unhandled** — returns list with non-float values | **Silent data corruption**: downstream cosine similarity computations produce NaN or errors |

### Critical Path Analysis

The vector flows through:

1. `embed_image()` → returns `list[float]` (line 61)
2. `generate_all_vectors()` in `pipeline.py` → stores in `ImageVectors.image_vector` (line 42)
3. `build_search_document()` in `indexer.py` → copies to `SearchDocument` fields (line 53)
4. `upload_documents()` → sends to Azure AI Search HNSW index

An empty or malformed vector at step 1 silently propagates through steps 2-3 because:

- `ImageVectors.image_vector` has `default_factory=list` — Pydantic accepts `[]` without complaint
- `SearchDocument.image_vector` also has `default_factory=list` — same issue
- Only at step 4 does Azure Search potentially reject it, but the error message won't trace back to Florence

For `embed_text_for_image_search()`, the vector flows through:

1. `embed_text_for_image_search()` → returns `list[float]`
2. `generate_query_vectors()` in `query.py` → added to `query_vectors["image_vector"]` (line 77)
3. `execute_hybrid_search()` in `search.py` → used in `VectorizedQuery` (line 85-91)

An empty vector at query time would produce a zero-weight vector query or potentially crash the Azure Search SDK.

---

## 4. Existing Error Handling Patterns in the Codebase

### Pattern 1: Input validation with `ValueError` (used consistently)

```python
# image.py L36-38
if not image_url and not image_bytes:
    msg = "Either image_url or image_bytes must be provided"
    raise ValueError(msg)

# extractor.py L73-75
if not image_url and not image_bytes:
    msg = "Either image_url or image_bytes must be provided"
    raise ValueError(msg)

# loader.py L38-40, metadata.py L46-48 — similar pattern
```

Convention: `msg = "..."` followed by `raise ValueError(msg)`.

### Pattern 2: HTTP error propagation via `raise_for_status()`

```python
# image.py L58, L86
response.raise_for_status()
```

No custom wrapping — `httpx.HTTPStatusError` propagates directly.

### Pattern 3: OpenAI SDK errors (implicit)

```python
# encoder.py L33-37
response = await _client.embeddings.create(
    model=config.models.embedding_model,
    input=chunk,
    dimensions=dimensions,
)
all_embeddings.extend([item.embedding for item in response.data])
```

The OpenAI SDK returns typed `Embedding` objects with validated `embedding: list[float]` fields. No manual response validation is needed because the SDK handles it. **This implicit safety does not apply to the Florence REST API.**

### Pattern 4: Retry with exponential backoff (indexer)

```python
# indexer.py L88-99
except HttpResponseError as e:
    if e.status_code in (429, 503) and attempt < max_retries - 1:
        delay = base_delay * (2**attempt)
```

Used for transient failures at the indexing layer.

### Summary

The codebase uses lightweight error handling: `ValueError` for input validation, implicit SDK validation for OpenAI, and `raise_for_status()` for HTTP errors. There is **no precedent** for response body validation or custom exception types. The simplest pattern that fits is `ValueError` with a descriptive message.

---

## 5. Validation Approaches

### Option A: Simple `data.get("vector")` with ValueError

```python
response.raise_for_status()
data = response.json()
vector = data.get("vector")
if not vector:
    msg = f"Florence API returned no vector: {data}"
    raise ValueError(msg)
```

**Pros:**

- Minimal code change (3 lines added)
- Consistent with existing `ValueError` pattern
- Covers both missing key and empty vector
- `not vector` handles `None`, `[]`, and missing key via `.get()`
- Includes response body in error message for debugging

**Cons:**

- Does not validate vector dimension or element types
- `f"...{data}"` could log sensitive data (unlikely for vectors)

### Option B: Pydantic response model validation

```python
from pydantic import BaseModel, Field, field_validator

class FlorenceResponse(BaseModel):
    vector: list[float] = Field(min_length=1)
    modelVersion: str | None = None

    @field_validator("vector")
    @classmethod
    def validate_dimensions(cls, v: list[float]) -> list[float]:
        if len(v) != 1024:
            msg = f"Expected 1024-dim vector, got {len(v)}"
            raise ValueError(msg)
        return v

# Usage:
response.raise_for_status()
parsed = FlorenceResponse.model_validate(response.json())
vector = parsed.vector
```

**Pros:**

- Full structural validation (key existence, type, dimension)
- Catches non-numeric values via `list[float]` coercion
- Self-documenting API contract
- Pydantic is already a dependency

**Cons:**

- Adds a model class for a simple two-field response
- Overhead of Pydantic validation per API call (~0.1ms, negligible vs network)
- More code than needed for the current risk level
- Inconsistent with the rest of the module's lightweight style

### Option C: Try/except with custom exception

```python
class FlorenceAPIError(Exception):
    """Raised when the Florence API returns an invalid response."""

# Usage:
response.raise_for_status()
try:
    data = response.json()
    vector: list[float] = data["vector"]
except (KeyError, json.JSONDecodeError) as e:
    raise FlorenceAPIError(f"Invalid Florence response: {e}") from e
if not vector:
    raise FlorenceAPIError("Florence returned empty vector")
```

**Pros:**

- Distinct exception type for Florence-specific errors
- Catches both JSON parse and key errors
- Clear chain of causation via `from e`

**Cons:**

- Introduces a custom exception class with no current codebase precedent
- More complex than needed
- Callers must now catch another exception type

### Option D: Hybrid — Simple validation + dimension check (RECOMMENDED)

```python
response.raise_for_status()
data = response.json()
vector = data.get("vector")
if not vector or len(vector) != 1024:
    msg = (
        f"Florence API response missing or invalid vector "
        f"(expected 1024-dim, got {len(vector) if vector else 'None'}): "
        f"keys={list(data.keys())}"
    )
    raise ValueError(msg)
```

**Pros:**

- Covers missing key, empty vector, AND wrong dimensions (most impactful silent corruption)
- Consistent with existing `ValueError` pattern
- Includes diagnostic info (actual length, response keys) without dumping full vector
- Minimal code addition (~5 lines per function)
- No new classes, imports, or exception types

**Cons:**

- Does not validate individual element types (unlikely failure mode)
- Hardcodes 1024 dimension (but this is already hardcoded as the Florence output dimension)

---

## 6. Recommended Approach

**Option D: Simple validation with dimension check using `ValueError`.**

### Rationale

1. **Consistency**: The codebase uses `ValueError` for input/output validation throughout. No custom exceptions exist.
2. **Coverage**: Catches the three most impactful failure modes: missing key (KeyError prevention), empty vector (silent corruption prevention), and wrong dimensions (index corruption prevention).
3. **Diagnostics**: Error message includes the actual vector length and response keys, making root-cause analysis straightforward.
4. **Minimal footprint**: ~5 lines per function, no new imports, no new classes.
5. **Pydantic is overkill**: The response has exactly one field we care about. A full model adds complexity without proportional benefit.

### Recommended Implementation

For `embed_image()` (lines 58-61):

```python
response.raise_for_status()
data = response.json()
vector = data.get("vector")
if not vector or len(vector) != 1024:
    msg = (
        f"Florence vectorizeImage response invalid "
        f"(expected 1024-dim vector, got {len(vector) if vector else 'None'}): "
        f"keys={list(data.keys())}"
    )
    raise ValueError(msg)

logger.info("Image embedded via Florence", dimensions=len(vector))
return vector
```

For `embed_text_for_image_search()` (lines 86-89):

```python
response.raise_for_status()
data = response.json()
vector = data.get("vector")
if not vector or len(vector) != 1024:
    msg = (
        f"Florence vectorizeText response invalid "
        f"(expected 1024-dim vector, got {len(vector) if vector else 'None'}): "
        f"keys={list(data.keys())}"
    )
    raise ValueError(msg)

logger.info("Text embedded via Florence (image space)", dimensions=len(vector))
return vector
```

### Docstring Updates

Add `ValueError` to the `Raises` section of `embed_text_for_image_search()` (currently undocumented) and update `embed_image()`'s docstring:

```python
Raises:
    ValueError: If neither image_url nor image_bytes is provided,
        or if the Florence API returns a missing or invalid vector.
    httpx.HTTPStatusError: If the Florence API returns an error HTTP status.
```

---

## 7. Required Test Updates

### Existing Tests That Need No Changes

- `test_embed_image_url` — tests happy path with valid `{"vector": [0.1] * 1024}`. Still valid.
- `test_embed_image_bytes` — tests happy path. Still valid.
- `test_embed_image_no_input_raises` — tests input validation. Still valid.
- `test_embed_image_api_error` — tests HTTP error propagation. Still valid.
- `test_embed_text_for_image_search` — tests happy path. Still valid.

### New Tests Required

Add to `tests/test_embeddings/test_image.py`:

```python
class TestEmbedImageResponseValidation:
    """Tests for Florence API response validation (IV-011)."""

    @pytest.mark.asyncio()
    async def test_embed_image_missing_vector_key(self, monkeypatch):
        """Should raise ValueError when response lacks 'vector' key."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.json.return_value = {"error": "something went wrong"}
        mock_response.raise_for_status = MagicMock()
        # ... setup client mock ...
        with pytest.raises(ValueError, match="missing or invalid vector"):
            await embed_image(image_url="https://example.com/test.jpg")

    @pytest.mark.asyncio()
    async def test_embed_image_empty_vector(self, monkeypatch):
        """Should raise ValueError when response vector is empty."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.json.return_value = {"vector": []}
        mock_response.raise_for_status = MagicMock()
        # ... setup client mock ...
        with pytest.raises(ValueError, match="missing or invalid vector"):
            await embed_image(image_url="https://example.com/test.jpg")

    @pytest.mark.asyncio()
    async def test_embed_image_wrong_dimensions(self, monkeypatch):
        """Should raise ValueError when vector has wrong dimension count."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.json.return_value = {"vector": [0.1] * 512}
        mock_response.raise_for_status = MagicMock()
        # ... setup client mock ...
        with pytest.raises(ValueError, match="expected 1024-dim"):
            await embed_image(image_url="https://example.com/test.jpg")

    @pytest.mark.asyncio()
    async def test_embed_text_missing_vector_key(self, monkeypatch):
        """Should raise ValueError when vectorizeText response lacks 'vector' key."""
        # Same pattern as above for embed_text_for_image_search
        ...

    @pytest.mark.asyncio()
    async def test_embed_text_empty_vector(self, monkeypatch):
        """Should raise ValueError when vectorizeText returns empty vector."""
        ...
```

### Test Count

- **Existing tests**: 5 (all remain valid, no modifications needed)
- **New tests needed**: 5-6 (3 for `embed_image`, 2-3 for `embed_text_for_image_search`)

---

## Summary

| Item | Detail |
|------|--------|
| **Finding** | IV-011: Unvalidated Florence API response access |
| **Severity** | Major — silent data corruption on empty/malformed vectors |
| **Root cause** | Direct `response.json()["vector"]` access without key/value checks |
| **Failure modes** | Missing key (`KeyError`), empty vector (silent corruption), wrong dimensions (index corruption), non-JSON body (`JSONDecodeError`) |
| **Recommended fix** | Option D: `data.get("vector")` + length check + `ValueError` |
| **Code impact** | ~10 lines added across 2 locations in `image.py` |
| **Test impact** | 5-6 new test cases; existing tests unchanged |
| **Risk** | Low — additive validation, no behavioral change for happy path |
