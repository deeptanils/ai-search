<!-- markdownlint-disable-file -->
# Implementation Details: Review Rework — Major Findings + Observability

## Context Reference

Sources:
* `.copilot-tracking/research/2026-02-26/review-rework-major-findings-research.md` — Consolidated research
* `.copilot-tracking/reviews/2026-02-26/multimodal-embeddings-plan-review.md` — Review log
* User request: "also consider observability and tracking"

## Implementation Phase 1: Secrets Hardening (IV-014)

<!-- parallelizable: false -->

### Step 1.1: Update `AzureComputerVisionSecrets` defaults to `str | None = None`

Change the default values from empty strings to `None` so that missing secrets are semantically distinct from empty values.

Files:
* `src/ai_search/config.py` — Modify `AzureComputerVisionSecrets` class (lines 60-72)

Changes:
```python
# BEFORE (lines 70-71)
endpoint: str = ""
api_key: str = ""

# AFTER
endpoint: str | None = None
api_key: str | None = None
```

Discrepancy references:
* Addresses IV-014 — empty string defaults mask misconfiguration

Success criteria:
* `AzureComputerVisionSecrets()` produces `endpoint=None`, `api_key=None` when env vars not set
* Other secrets classes unchanged

Context references:
* `src/ai_search/config.py` (Lines 60-72) — `AzureComputerVisionSecrets` class
* Research doc, Scenario 3 — Selected approach: Option D

Dependencies:
* None — first step

### Step 1.2: Add validation guard with structured logging in `get_cv_client()`

Add a validation check at the top of `get_cv_client()` that raises `ValueError` with an actionable error message when `endpoint` or `api_key` is `None`. Log the failure via `structlog` before raising.

Files:
* `src/ai_search/clients.py` — Modify `get_cv_client()` function (lines 73-79)

Changes:
```python
# Add structlog import at the top of clients.py
import structlog

logger = structlog.get_logger(__name__)

# BEFORE (lines 73-79)
@lru_cache(maxsize=1)
def get_cv_client() -> httpx.AsyncClient:
    """Return a cached async HTTP client for Azure Computer Vision."""
    secrets = load_cv_secrets()
    return httpx.AsyncClient(
        base_url=secrets.endpoint,
        headers={"Ocp-Apim-Subscription-Key": secrets.api_key},
        timeout=30.0,
    )

# AFTER
@lru_cache(maxsize=1)
def get_cv_client() -> httpx.AsyncClient:
    """Return a cached async HTTP client for Azure Computer Vision."""
    secrets = load_cv_secrets()
    if not secrets.endpoint or not secrets.api_key:
        logger.error(
            "Azure Computer Vision secrets not configured",
            endpoint_set=secrets.endpoint is not None,
            api_key_set=secrets.api_key is not None,
        )
        msg = (
            "Azure Computer Vision secrets not configured. "
            "Set AZURE_CV_ENDPOINT and AZURE_CV_API_KEY environment variables."
        )
        raise ValueError(msg)
    return httpx.AsyncClient(
        base_url=secrets.endpoint,
        headers={"Ocp-Apim-Subscription-Key": secrets.api_key},
        timeout=30.0,
    )
```

Note: After the guard, `secrets.endpoint` and `secrets.api_key` are guaranteed non-None `str`, satisfying `httpx.AsyncClient` type expectations. The `@lru_cache` on `get_cv_client()` means the validation runs once per process; subsequent calls use the cached client.

Discrepancy references:
* Addresses IV-014 — validation guard prevents broken client construction
* Addresses observability — `logger.error()` for secrets-guard failure

Success criteria:
* `get_cv_client()` raises `ValueError` when endpoint or api_key is `None`
* Error message names exact env vars to set
* `logger.error()` emits structured log before raising

Context references:
* `src/ai_search/clients.py` (Lines 73-79) — current `get_cv_client()`
* Research doc, Scenario 3 — validation guard implementation

Dependencies:
* Step 1.1 (secrets type change must happen first)

## Implementation Phase 2: Response Validation + Observability (IV-011)

<!-- parallelizable: false -->

### Step 2.1: Add response validation with structured logging to `embed_image()`

Replace the direct `response.json()["vector"]` access with validated extraction: `data.get("vector")` + dimension check + `ValueError` with diagnostic info. Add `logger.warning()` before raising to capture validation failures in structured logs.

Files:
* `src/ai_search/embeddings/image.py` — Modify `embed_image()` function (lines 59-64)

Changes:
```python
# BEFORE (lines 61-64)
    response.raise_for_status()
    vector: list[float] = response.json()["vector"]

    logger.info("Image embedded via Florence", dimensions=len(vector))
    return vector

# AFTER
    response.raise_for_status()
    data = response.json()
    vector = data.get("vector")
    if not vector or len(vector) != 1024:
        logger.warning(
            "Florence vectorizeImage response invalid",
            expected_dimensions=1024,
            actual_dimensions=len(vector) if vector else None,
            response_keys=list(data.keys()),
        )
        msg = (
            f"Florence vectorizeImage response invalid "
            f"(expected 1024-dim vector, got {len(vector) if vector else 'None'}): "
            f"keys={list(data.keys())}"
        )
        raise ValueError(msg)

    logger.info("Image embedded via Florence", dimensions=len(vector))
    return vector
```

Discrepancy references:
* Addresses IV-011 — validates response payload
* Addresses observability — `logger.warning()` captures invalid responses with structured fields

Success criteria:
* `embed_image()` raises `ValueError` on missing `vector` key, empty vector, or wrong dimensions
* Warning log emitted with `expected_dimensions`, `actual_dimensions`, `response_keys`
* Happy path unchanged (still returns `list[float]` and logs `info`)

Context references:
* `src/ai_search/embeddings/image.py` (Lines 59-64) — current response handling
* Research doc, Scenario 2 — validation pattern

Dependencies:
* Phase 1 complete (secrets change doesn't affect these lines, but sequential for safety)

### Step 2.2: Add response validation with structured logging to `embed_text_for_image_search()`

Same validation pattern as Step 2.1, applied to the `vectorizeText` path.

Files:
* `src/ai_search/embeddings/image.py` — Modify `embed_text_for_image_search()` function (lines 89-94)

Changes:
```python
# BEFORE (lines 91-94)
    response.raise_for_status()
    vector: list[float] = response.json()["vector"]

    logger.info("Text embedded via Florence (image space)", dimensions=len(vector))
    return vector

# AFTER
    response.raise_for_status()
    data = response.json()
    vector = data.get("vector")
    if not vector or len(vector) != 1024:
        logger.warning(
            "Florence vectorizeText response invalid",
            expected_dimensions=1024,
            actual_dimensions=len(vector) if vector else None,
            response_keys=list(data.keys()),
        )
        msg = (
            f"Florence vectorizeText response invalid "
            f"(expected 1024-dim vector, got {len(vector) if vector else 'None'}): "
            f"keys={list(data.keys())}"
        )
        raise ValueError(msg)

    logger.info("Text embedded via Florence (image space)", dimensions=len(vector))
    return vector
```

Discrepancy references:
* Addresses IV-011 — both embedding functions now validate
* Addresses observability — `logger.warning()` for text-to-image validation failures

Success criteria:
* `embed_text_for_image_search()` raises `ValueError` on invalid responses
* Structured warning log before raise
* Happy path unchanged

Context references:
* `src/ai_search/embeddings/image.py` (Lines 89-94) — current response handling

Dependencies:
* Step 2.1 (same file, sequential edits)

### Step 2.3: Add 6 validation tests to `test_image.py`

Add tests verifying that invalid Florence responses are caught. Use the existing `monkeypatch` pattern from the file's `mock_cv_client` fixture as a template, with modified response payloads.

Files:
* `tests/test_embeddings/test_image.py` — Add new test classes after existing tests

New tests:

```python
class TestEmbedImageValidation:
    """Test embed_image response validation."""

    @pytest.mark.asyncio()
    async def test_missing_vector_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should raise ValueError when response lacks 'vector' key."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.json.return_value = {"modelVersion": "2023-04-15"}
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        secrets = MagicMock()
        secrets.api_version = "2024-02-01"
        monkeypatch.setattr("ai_search.embeddings.image.get_cv_client", lambda: mock_client)
        monkeypatch.setattr("ai_search.embeddings.image.load_cv_secrets", lambda: secrets)

        with pytest.raises(ValueError, match="Florence vectorizeImage response invalid"):
            await embed_image(image_url="https://example.com/test.jpg")

    @pytest.mark.asyncio()
    async def test_empty_vector_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should raise ValueError when response vector is empty."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.json.return_value = {"vector": []}
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        secrets = MagicMock()
        secrets.api_version = "2024-02-01"
        monkeypatch.setattr("ai_search.embeddings.image.get_cv_client", lambda: mock_client)
        monkeypatch.setattr("ai_search.embeddings.image.load_cv_secrets", lambda: secrets)

        with pytest.raises(ValueError, match="Florence vectorizeImage response invalid"):
            await embed_image(image_url="https://example.com/test.jpg")

    @pytest.mark.asyncio()
    async def test_wrong_dimensions_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should raise ValueError when vector has wrong dimensions."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.json.return_value = {"vector": [0.1] * 512}
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        secrets = MagicMock()
        secrets.api_version = "2024-02-01"
        monkeypatch.setattr("ai_search.embeddings.image.get_cv_client", lambda: mock_client)
        monkeypatch.setattr("ai_search.embeddings.image.load_cv_secrets", lambda: secrets)

        with pytest.raises(ValueError, match="expected 1024-dim vector, got 512"):
            await embed_image(image_url="https://example.com/test.jpg")


class TestEmbedTextForImageSearchValidation:
    """Test embed_text_for_image_search response validation."""

    @pytest.mark.asyncio()
    async def test_missing_vector_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should raise ValueError when response lacks 'vector' key."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.json.return_value = {"error": "unexpected"}
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        secrets = MagicMock()
        secrets.api_version = "2024-02-01"
        monkeypatch.setattr("ai_search.embeddings.image.get_cv_client", lambda: mock_client)
        monkeypatch.setattr("ai_search.embeddings.image.load_cv_secrets", lambda: secrets)

        with pytest.raises(ValueError, match="Florence vectorizeText response invalid"):
            await embed_text_for_image_search("sunset over mountains")

    @pytest.mark.asyncio()
    async def test_wrong_dimensions_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should raise ValueError when vector has wrong dimensions."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.json.return_value = {"vector": [0.1] * 256}
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        secrets = MagicMock()
        secrets.api_version = "2024-02-01"
        monkeypatch.setattr("ai_search.embeddings.image.get_cv_client", lambda: mock_client)
        monkeypatch.setattr("ai_search.embeddings.image.load_cv_secrets", lambda: secrets)

        with pytest.raises(ValueError, match="expected 1024-dim vector, got 256"):
            await embed_text_for_image_search("sunset over mountains")


class TestSecretsValidation:
    """Test secrets misconfiguration detection."""

    @pytest.mark.asyncio()
    async def test_missing_secrets_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should raise ValueError when CV secrets are not configured."""
        from ai_search.clients import get_cv_client

        secrets = MagicMock()
        secrets.endpoint = None
        secrets.api_key = None
        monkeypatch.setattr("ai_search.clients.load_cv_secrets", lambda: secrets)
        get_cv_client.cache_clear()

        with pytest.raises(ValueError, match="Azure Computer Vision secrets not configured"):
            get_cv_client()
```

Discrepancy references:
* Addresses IV-011 — validates all three failure modes (missing key, empty, wrong dims)
* Addresses IV-014 — tests secrets validation guard

Success criteria:
* 6 new tests pass (3 for embed_image, 2 for embed_text_for_image_search, 1 for secrets)
* All 5 existing tests still pass
* Tests use match patterns to verify error messages

Context references:
* `tests/test_embeddings/test_image.py` (Lines 1-99) — existing test patterns

Dependencies:
* Steps 2.1, 2.2 complete (validation logic must exist for tests to pass)
* Step 1.2 complete (secrets guard must exist for misconfiguration test)

## Implementation Phase 3: Missing Query Tests (RPI-004)

<!-- parallelizable: true -->

### Step 3.1: Create `tests/test_retrieval/test_query.py` with 3 tests

Create the missing test file with a `TestGenerateQueryVectors` class containing 3 async tests. Use stacked `@patch()` decorators matching `test_search.py` conventions.

Files:
* `tests/test_retrieval/test_query.py` — NEW file

Full file content:

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
    async def test_returns_all_vector_keys(
        self,
        mock_config: MagicMock,
        mock_client: MagicMock,
        mock_embed_text: AsyncMock,
        mock_embed_text_image: AsyncMock,
    ) -> None:
        """Should return dict with all four vector keys including image_vector."""
        from ai_search.config import AppConfig

        mock_config.return_value = AppConfig()

        # LLM responses (sync)
        llm_response = MagicMock()
        llm_response.choices = [MagicMock()]
        llm_response.choices[0].message.content = "test description"
        mock_client.return_value.chat.completions.create.return_value = llm_response

        # embed_text returns dimension-aware vectors
        mock_embed_text.side_effect = lambda text, dimensions: [0.1] * dimensions

        # embed_text_for_image_search returns 1024-dim
        mock_embed_text_image.return_value = [0.2] * 1024

        result = await generate_query_vectors("sunset photo")

        assert set(result.keys()) == {
            "semantic_vector",
            "structural_vector",
            "style_vector",
            "image_vector",
        }
        assert len(result["semantic_vector"]) == 3072
        assert len(result["structural_vector"]) == 1024
        assert len(result["style_vector"]) == 512
        assert len(result["image_vector"]) == 1024

    @pytest.mark.asyncio()
    @patch("ai_search.retrieval.query.embed_text_for_image_search", new_callable=AsyncMock)
    @patch("ai_search.retrieval.query.embed_image", new_callable=AsyncMock)
    @patch("ai_search.retrieval.query.embed_text", new_callable=AsyncMock)
    @patch("ai_search.retrieval.query.get_openai_client")
    @patch("ai_search.retrieval.query.load_config")
    async def test_image_url_calls_embed_image(
        self,
        mock_config: MagicMock,
        mock_client: MagicMock,
        mock_embed_text: AsyncMock,
        mock_embed_image: AsyncMock,
        mock_embed_text_image: AsyncMock,
    ) -> None:
        """Should call embed_image when query_image_url is provided."""
        from ai_search.config import AppConfig

        mock_config.return_value = AppConfig()

        llm_response = MagicMock()
        llm_response.choices = [MagicMock()]
        llm_response.choices[0].message.content = "test description"
        mock_client.return_value.chat.completions.create.return_value = llm_response

        mock_embed_text.side_effect = lambda text, dimensions: [0.1] * dimensions
        mock_embed_image.return_value = [0.3] * 1024

        result = await generate_query_vectors(
            "sunset photo",
            query_image_url="https://example.com/ref.jpg",
        )

        mock_embed_image.assert_called_once_with(image_url="https://example.com/ref.jpg")
        mock_embed_text_image.assert_not_called()
        assert len(result["image_vector"]) == 1024

    @pytest.mark.asyncio()
    @patch("ai_search.retrieval.query.embed_image", new_callable=AsyncMock)
    @patch("ai_search.retrieval.query.embed_text_for_image_search", new_callable=AsyncMock)
    @patch("ai_search.retrieval.query.embed_text", new_callable=AsyncMock)
    @patch("ai_search.retrieval.query.get_openai_client")
    @patch("ai_search.retrieval.query.load_config")
    async def test_text_only_calls_embed_text_for_image_search(
        self,
        mock_config: MagicMock,
        mock_client: MagicMock,
        mock_embed_text: AsyncMock,
        mock_embed_text_image: AsyncMock,
        mock_embed_image: AsyncMock,
    ) -> None:
        """Should call embed_text_for_image_search when no image URL provided."""
        from ai_search.config import AppConfig

        mock_config.return_value = AppConfig()

        llm_response = MagicMock()
        llm_response.choices = [MagicMock()]
        llm_response.choices[0].message.content = "test description"
        mock_client.return_value.chat.completions.create.return_value = llm_response

        mock_embed_text.side_effect = lambda text, dimensions: [0.1] * dimensions
        mock_embed_text_image.return_value = [0.4] * 1024

        result = await generate_query_vectors("sunset photo")

        mock_embed_text_image.assert_called_once_with("sunset photo")
        mock_embed_image.assert_not_called()
        assert len(result["image_vector"]) == 1024
```

Discrepancy references:
* Addresses RPI-004 — creates the missing test file

Success criteria:
* File exists at `tests/test_retrieval/test_query.py`
* 3 tests pass: `test_returns_all_vector_keys`, `test_image_url_calls_embed_image`, `test_text_only_calls_embed_text_for_image_search`
* Tests verify branch exclusivity (embed_image NOT called in text-only path and vice versa)

Context references:
* `src/ai_search/retrieval/query.py` (Lines 31-84) — function under test
* Research doc, Scenario 1 — test structure and mock patterns

Dependencies:
* None — independent file, can run in parallel with Phase 2

## Implementation Phase 4: Validation

<!-- parallelizable: false -->

### Step 4.1: Run full project validation

Execute all validation commands for the project:
* `uv run ruff check .`
* `uv run mypy src/`
* `uv run pytest -v` — expect 63+ tests (57 existing + 6 new)

### Step 4.2: Fix minor validation issues

Iterate on lint errors, build warnings, and test failures. Apply fixes directly when corrections are straightforward and isolated.

### Step 4.3: Report blocking issues

When validation failures require changes beyond minor fixes:
* Document the issues and affected files
* Provide the user with next steps
* Recommend additional research and planning rather than inline fixes
* Avoid large-scale refactoring within this phase

## Dependencies

* Python 3.12 with UV package manager
* structlog (already installed)
* httpx, pytest, pytest-asyncio, ruff, mypy (already installed)

## Success Criteria

* All 3 major findings (RPI-004, IV-011, IV-014) addressed
* Observability: structured warning/error logs on all new error paths
* 63+ tests pass with zero regressions
* ruff, mypy clean
