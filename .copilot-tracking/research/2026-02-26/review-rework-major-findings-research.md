<!-- markdownlint-disable-file -->
# Task Research: Review Rework — Major Findings

Research for addressing the 3 major findings from the multimodal image embeddings review.

## Task Implementation Requests

* RPI-004: Create missing `tests/test_retrieval/test_query.py` with tests for `generate_query_vectors()` image embedding paths
* IV-011: Add Florence API response payload validation to prevent `KeyError` and silent data corruption
* IV-014: Fix `AzureComputerVisionSecrets` empty string defaults to fail fast on missing credentials

## Scope and Success Criteria

* Scope: Rework of 3 major review findings (RPI-004, IV-011, IV-014). Source files limited to `src/ai_search/embeddings/image.py`, `src/ai_search/config.py`, `src/ai_search/clients.py`, new `tests/test_retrieval/test_query.py`, and updated `tests/test_embeddings/test_image.py`. Does not include minor findings (IV-005, IV-009, IV-012, IV-013, IV-018-020, IV-026) or deferred follow-on work (WI-01 through WI-08).
* Assumptions:
  * Florence API is opt-in (only runs when image_url or image_bytes is provided)
  * Existing test patterns (monkeypatch, pytest-asyncio, conftest fixtures) should be followed
  * Changes must pass ruff, mypy, and pytest validation
  * Florence always returns 1024-dim vectors in GA
* Success Criteria:
  * `tests/test_retrieval/test_query.py` exists with 3 tests covering image_vector key, image URL path, and text-only path
  * `embed_image()` and `embed_text_for_image_search()` validate response payloads (key existence, non-empty, 1024 dims)
  * `AzureComputerVisionSecrets` uses `str | None = None` instead of `str = ""` with validation guard in `get_cv_client()`
  * All findings resolved with zero regressions in existing 57 tests
  * ruff, mypy, and pytest pass clean

## Outline

1. RPI-004: Missing test_query.py — function signatures, mock structure, test patterns
2. IV-011: Florence response validation — API schema, failure modes, validation approach
3. IV-014: Secrets defaults — existing patterns, opt-in vs required, validation guard
4. Technical scenarios with selected approaches

## Potential Next Research

* IV-012 (Minor): Input validation for `embed_text_for_image_search()` empty text — straightforward `ValueError` check, can be bundled with IV-011 fix
  * Reasoning: Same file, same function, minimal additional effort
  * Reference: Implementation validation log IV-012

## Research Executed

### File Analysis

* `src/ai_search/retrieval/query.py` (Lines 1-93)
  * `generate_query_vectors()` is async, returns `dict[str, list[float]]` with keys: semantic_vector, structural_vector, style_vector, image_vector
  * 5 dependencies to mock: `load_config`, `get_openai_client`, `embed_text`, `embed_image`, `embed_text_for_image_search`
  * Branch: `query_image_url` provided → `embed_image()`; absent → `embed_text_for_image_search()`
  * LLM client is synchronous (`chat.completions.create()` not awaited)

* `src/ai_search/embeddings/image.py` (Lines 1-93)
  * `embed_image()` and `embed_text_for_image_search()` both access `response.json()["vector"]` without validation
  * `raise_for_status()` covers HTTP errors only
  * Three unprotected failure points: JSONDecodeError, KeyError, empty/wrong-dim vector
  * Florence returns `{"vector": [...], "modelVersion": "..."}` — vector is always 1024-dim

* `src/ai_search/config.py` (Lines 58-72)
  * `AzureComputerVisionSecrets` has `endpoint: str = ""` and `api_key: str = ""`
  * Other secrets classes (`AzureFoundrySecrets`, `AzureSearchSecrets`) have no defaults → fail fast
  * `load_cv_secrets()` is lazy (called only when Florence is used), cached via `@lru_cache`

* `src/ai_search/clients.py` (Lines 67-79)
  * `get_cv_client()` uses `secrets.endpoint` as `base_url` — empty string creates broken client

### Code Search Results

* `load_cv_secrets` — 3 callers: `get_cv_client()` in clients.py, `embed_image()` and `embed_text_for_image_search()` in image.py
* `get_cv_client` — 2 callers: both in image.py
* All callers are lazy (guarded behind `has_image` check in pipeline.py)

### Project Conventions

* Standards referenced: `requirements.md` Section 12, commit message instructions
* Test conventions: `@pytest.mark.asyncio()`, stacked `@patch()` decorators, `new_callable=AsyncMock` for async targets, `MagicMock` for sync
* Error handling convention: `ValueError` with `msg = "..."` pattern, no custom exception classes
* Secrets convention: Required services have no defaults (fail fast); optional services use defaults

## Key Discoveries

### Project Structure

The codebase follows a clean dependency graph: `config → clients → embeddings → models → indexing → retrieval`. Test files mirror source structure under `tests/`. Each retrieval module should have a corresponding test file.

### Implementation Patterns

* **Mock pattern**: `@patch("module.path.function", new_callable=AsyncMock)` for async; `@patch("module.path.function")` for sync. Decorators stacked in reverse parameter order.
* **Error pattern**: `msg = "descriptive message"` followed by `raise ValueError(msg)`. No custom exception types in codebase.
* **Secrets pattern**: Required services use bare `str` (no default); optional services can use `str | None = None` with lazy validation.
* **Response validation**: OpenAI SDK validates responses implicitly. Florence REST API has no SDK validation — must be done manually.

### Complete Examples

#### Test structure for `test_query.py`

```python
"""Tests for query vector generation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_search.retrieval.query import generate_query_vectors


class TestGenerateQueryVectors:
    """Test the generate_query_vectors function."""

    @pytest.mark.asyncio()
    @patch("ai_search.retrieval.query.embed_text_for_image_search", new_callable=AsyncMock)
    @patch("ai_search.retrieval.query.embed_text", new_callable=AsyncMock)
    @patch("ai_search.retrieval.query.get_openai_client")
    @patch("ai_search.retrieval.query.load_config")
    async def test_returns_image_vector_key(
        self,
        mock_config: MagicMock,
        mock_client: MagicMock,
        mock_embed_text: AsyncMock,
        mock_embed_text_image: AsyncMock,
    ) -> None:
        # Setup mocks, call function, assert all 4 keys present
        ...

    @pytest.mark.asyncio()
    @patch("ai_search.retrieval.query.embed_image", new_callable=AsyncMock)
    @patch("ai_search.retrieval.query.embed_text", new_callable=AsyncMock)
    @patch("ai_search.retrieval.query.get_openai_client")
    @patch("ai_search.retrieval.query.load_config")
    async def test_image_url_calls_embed_image(self, ...): ...

    @pytest.mark.asyncio()
    @patch("ai_search.retrieval.query.embed_text_for_image_search", new_callable=AsyncMock)
    @patch("ai_search.retrieval.query.embed_text", new_callable=AsyncMock)
    @patch("ai_search.retrieval.query.get_openai_client")
    @patch("ai_search.retrieval.query.load_config")
    async def test_text_only_calls_embed_text_for_image_search(self, ...): ...
```

#### Florence response validation pattern

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
```

#### Secrets with None defaults

```python
endpoint: str | None = None
api_key: str | None = None
```

### API and Schema Documentation

* Florence `vectorizeImage` endpoint: POST `/computervision/retrieval:vectorizeImage`
  * Success: `{"vector": [float * 1024], "modelVersion": "2023-04-15"}`
  * Error: `{"error": {"code": "...", "message": "...", "innererror": {...}}}`
* Florence `vectorizeText` endpoint: POST `/computervision/retrieval:vectorizeText`
  * Same response schema as vectorizeImage

## Technical Scenarios

### Scenario 1: RPI-004 — Create test_query.py

Missing test file for `generate_query_vectors()` image embedding code paths.

**Requirements:**

* 3 tests covering: return dict keys, image URL branch, text-only branch
* Follow existing test patterns (stacked @patch decorators, pytest-asyncio)
* Mock 5 dependencies at the module boundary

**Preferred Approach:**

* Single test class `TestGenerateQueryVectors` with 3 async test methods
* Stacked `@patch()` decorators matching `test_pipeline.py` conventions
* `embed_text` mocked with `side_effect=lambda text, dimensions: [0.1] * dimensions` for dimension-aware returns
* Branch exclusivity asserted: Test 2 verifies `embed_text_for_image_search` NOT called; Test 3 verifies `embed_image` NOT called

```text
tests/test_retrieval/
  test_query.py  (NEW — 3 tests)
```

**Implementation Details:**

* LLM client mock must be synchronous (`MagicMock` not `AsyncMock`) — `client.chat.completions.create()` is not awaited
* `embed_text` is called 3 times with different dimension kwargs (3072, 1024, 512) — use `side_effect` for correct return sizes
* `load_config` returns `AppConfig()` with defaults (dimensions already configured)
* No fixtures needed — all setup inline via `@patch` decorators

#### Considered Alternatives

* **Using conftest fixtures instead of @patch**: Rejected. `test_search.py` (same module) uses inline @patch; consistency preferred.
* **Testing `generate_query_vectors_sync` too**: Not required by RPI-004; sync wrapper is a simple `asyncio.run()` call.

### Scenario 2: IV-011 — Florence Response Validation

Unvalidated `response.json()["vector"]` access causes KeyError or silent data corruption.

**Requirements:**

* Validate response key existence, non-emptiness, and correct dimension (1024)
* Use existing `ValueError` pattern
* Add tests for missing key, empty vector, and wrong dimensions

**Preferred Approach:**

* Option D: Simple `data.get("vector")` + dimension check + `ValueError`
* Apply to both `embed_image()` and `embed_text_for_image_search()`
* ~5 lines per function, no new imports or exception classes
* 5 new tests in `test_image.py` (3 for embed_image, 2 for embed_text_for_image_search)

```text
src/ai_search/embeddings/
  image.py  (MODIFIED — add validation after raise_for_status)
tests/test_embeddings/
  test_image.py  (MODIFIED — add 5 validation tests)
```

**Implementation Details:**

Error message includes diagnostic info (`len(vector)` and `list(data.keys())`) without dumping the full vector. Hardcoded 1024 dimension matches Florence GA model output. The `not vector` check handles both `None` (missing key → `.get()` returns `None`) and `[]` (empty list is falsy).

#### Considered Alternatives

* **Option A (simple get without dimension check)**: Rejected. Misses wrong-dimension corruption — the most dangerous silent failure.
* **Option B (Pydantic response model)**: Rejected. Overkill for a single-field response; adds a model class without proportional benefit.
* **Option C (custom FlorenceAPIError exception)**: Rejected. No precedent for custom exceptions in codebase; callers would need to catch a new type.

### Scenario 3: IV-014 — Secrets Empty String Defaults

`AzureComputerVisionSecrets` silently accepts empty endpoint/api_key, creating broken httpx client.

**Requirements:**

* Fail with clear error when Florence is used without configuration
* Preserve Florence opt-in nature (no env vars needed if image embeddings unused)
* Match codebase secrets conventions

**Preferred Approach:**

* Option D: Change `str = ""` to `str | None = None` + validation guard in `get_cv_client()`
* `None` is the canonical Python "not provided" sentinel
* Validation is lazy — only triggers when Florence is actually used
* Error message names the exact env variables to set
* 1 new test for misconfiguration error

```text
src/ai_search/
  config.py   (MODIFIED — change defaults to None)
  clients.py  (MODIFIED — add validation guard)
tests/test_embeddings/
  test_image.py  (MODIFIED — add misconfiguration test) OR
tests/test_config.py  (MODIFIED — add misconfiguration test)
```

**Implementation Details:**

The `str | None` type requires adding `| None` to the `httpx.AsyncClient` constructor call (`base_url` expects `str`). The validation guard in `get_cv_client()` resolves this by raising before the client is constructed. `@lru_cache` on `get_cv_client()` means the validation runs once; subsequent calls use the cached client. The `lru_cache` must be cleared in the misconfiguration test to avoid interference.

#### Considered Alternatives

* **Option A (remove defaults entirely)**: Rejected. Breaks users who never use Florence — secrets would fail at load time even when image embeddings aren't needed.
* **Option B (add `enabled: bool = False` flag)**: Rejected. Overengineered; adds a configuration concept not used elsewhere.
* **Option C (validate in get_cv_client only, keep empty strings)**: Rejected. Empty strings remain valid at the model level, which is semantically wrong.
