---
title: "RPI-004 Research: test_query.py for generate_query_vectors()"
description: "Subagent research document for creating missing tests/test_retrieval/test_query.py"
ms.date: 2026-02-26
---

## Summary

RPI-004 (Major) requires creating `tests/test_retrieval/test_query.py` with 3 tests for `generate_query_vectors()`. This document captures complete research on the source function, mocking requirements, existing test conventions, and recommended test structure.

## Source function analysis

### File: `src/ai_search/retrieval/query.py`

#### Complete function signature

```python
async def generate_query_vectors(
    query_text: str,
    query_image_url: str | None = None,
) -> dict[str, list[float]]:
```

#### Sync wrapper

```python
def generate_query_vectors_sync(
    query_text: str,
    query_image_url: str | None = None,
) -> dict[str, list[float]]:
    """Synchronous wrapper for query vector generation."""
    return asyncio.run(generate_query_vectors(query_text, query_image_url))
```

#### Return value structure

```python
{
    "semantic_vector": list[float],    # embed_text(query_text, dims.semantic)
    "structural_vector": list[float],  # embed_text(structural_desc, dims.structural)
    "style_vector": list[float],       # embed_text(style_desc, dims.style)
    "image_vector": list[float],       # embed_image() OR embed_text_for_image_search()
}
```

#### Import dependencies (mock targets)

| Import | Module path | Purpose |
|--------|-------------|---------|
| `get_openai_client` | `ai_search.clients` | Synchronous OpenAI client for LLM chat completions |
| `load_config` | `ai_search.config` | Loads `AppConfig` with dimension settings |
| `embed_text` | `ai_search.embeddings.encoder` | Text embedding via OpenAI |
| `embed_image` | `ai_search.embeddings.image` | Image embedding via Florence (URL path) |
| `embed_text_for_image_search` | `ai_search.embeddings.image` | Text-to-image embedding via Florence (text path) |

### Code paths requiring coverage

| # | Path | Condition | Called function |
|---|------|-----------|-----------------|
| 1 | Image URL provided | `query_image_url` is truthy | `embed_image(image_url=query_image_url)` |
| 2 | Text-only query | `query_image_url` is `None` | `embed_text_for_image_search(query_text)` |

Both paths share the common prefix: LLM generates structural and style descriptions, then `embed_text()` is called 3 times in parallel via `asyncio.gather()`.

## Mock and patch requirements

### Patches needed (all on `ai_search.retrieval.query` module path)

| Target | Mock type | Return value |
|--------|-----------|--------------|
| `ai_search.retrieval.query.load_config` | `MagicMock` | `AppConfig()` (default config works) |
| `ai_search.retrieval.query.get_openai_client` | `MagicMock` | Mock client with `chat.completions.create()` returning a mock with `choices[0].message.content = "test description"` |
| `ai_search.retrieval.query.embed_text` | `AsyncMock` | Dynamic return based on `dimensions` kwarg: `[0.1] * dimensions` |
| `ai_search.retrieval.query.embed_image` | `AsyncMock` | `[0.4] * 1024` |
| `ai_search.retrieval.query.embed_text_for_image_search` | `AsyncMock` | `[0.5] * 1024` |

### LLM client mock detail

The function calls `client.chat.completions.create()` twice (structural, style). The mock must return an object with `choices[0].message.content` as a string. The existing `mock_openai_client` fixture in `conftest.py` already sets this up correctly — it returns `"test response"` for `client.chat.completions.create()`.

### Async considerations

* `generate_query_vectors` is `async`; tests must use `@pytest.mark.asyncio()` decorator.
* `embed_text`, `embed_image`, and `embed_text_for_image_search` are awaited inside the function, so they must be patched with `AsyncMock` (using `new_callable=AsyncMock`).
* The LLM client (`get_openai_client`) is synchronous — it returns a regular `MagicMock`.

## Existing test patterns and conventions

### Conventions observed across the test suite

1. **Module docstring**: Always present, describes the test module.
2. **Test class**: Named `TestFunctionName` (PascalCase), groups related tests.
3. **Decorators**: `@pytest.mark.asyncio()` for async tests, `@patch(...)` for mocking (with `new_callable=AsyncMock` for async targets).
4. **Type annotations**: All test method parameters are annotated, return type is `-> None`.
5. **Imports**: `from __future__ import annotations` at top; imports from `unittest.mock` for `AsyncMock`, `MagicMock`, `patch`.
6. **Assertion style**: Direct `assert` statements (no `self.assert*`), `assert_called_once()`, `assert_called_once_with()`.
7. **Mock setup**: Inline `mock.return_value = ...` rather than fixtures for most patches.
8. **Config mocking**: `AppConfig()` instantiated with defaults via `from ai_search.config import AppConfig`.

### Pattern from `test_pipeline.py` (closest analog)

The pipeline test uses stacked `@patch()` decorators with `new_callable=AsyncMock` for async embedding functions. This is the exact pattern needed for `test_query.py`.

```python
@pytest.mark.asyncio()
@patch("ai_search.embeddings.pipeline.generate_style_vector", new_callable=AsyncMock)
@patch("ai_search.embeddings.pipeline.generate_structural_vector", new_callable=AsyncMock)
@patch("ai_search.embeddings.pipeline.generate_semantic_vector", new_callable=AsyncMock)
async def test_parallel_execution(
    self,
    mock_semantic: AsyncMock,
    mock_structural: AsyncMock,
    mock_style: AsyncMock,
    ...
) -> None:
```

### Pattern from `test_search.py` (same retrieval module)

Uses `@patch()` with `MagicMock` for synchronous dependencies (`get_search_client`, `load_config`). Config is set up as `AppConfig()` with defaults.

### Pattern from `test_image.py` (Florence mocking)

Uses `monkeypatch.setattr` for Florence client mocking. However, for `test_query.py` we do not need to mock Florence directly—we mock the higher-level `embed_image` and `embed_text_for_image_search` functions at the `ai_search.retrieval.query` module boundary.

## Plan specification (Step 6.3)

From `.copilot-tracking/details/2026-02-26/multimodal-embeddings-details.md` (lines 735-745):

> Query test:
>
> * Mock LLM + embedding + Florence clients
> * Verify `generate_query_vectors()` returns dict with `image_vector` key
> * Verify image-only query calls `embed_image()` instead of `embed_text_for_image_search()`

The plan says "image-only query calls `embed_image()`" but the actual code path is when `query_image_url` is provided (not "image-only"). The third test from the review finding (RPI-004) specifies: "text-only queries call `embed_text_for_image_search()`".

### Plan vs implementation alignment

| Plan spec | Actual need |
|-----------|-------------|
| "Mock LLM + embedding + Florence clients" | Mock `get_openai_client`, `load_config`, `embed_text`, `embed_image`, `embed_text_for_image_search` |
| "returns dict with `image_vector` key" | Test 1: assert `"image_vector"` in result dict |
| "image URL query calls `embed_image()`" | Test 2: pass `query_image_url`, assert `embed_image` called, `embed_text_for_image_search` not called |
| (from RPI-004 review finding) "text-only queries call `embed_text_for_image_search()`" | Test 3: omit `query_image_url`, assert `embed_text_for_image_search` called, `embed_image` not called |

## Recommended test structure

### File: `tests/test_retrieval/test_query.py`

```python
"""Tests for query vector generation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_search.retrieval.query import generate_query_vectors


class TestGenerateQueryVectors:
    """Test the generate_query_vectors function."""

    # Test 1: Verify return dict includes image_vector key
    @pytest.mark.asyncio()
    @patch("ai_search.retrieval.query.embed_text_for_image_search", new_callable=AsyncMock)
    @patch("ai_search.retrieval.query.embed_text", new_callable=AsyncMock)
    @patch("ai_search.retrieval.query.get_openai_client")
    @patch("ai_search.retrieval.query.load_config")
    async def test_returns_image_vector_key(self, ...): ...

    # Test 2: Verify image URL queries call embed_image()
    @pytest.mark.asyncio()
    @patch("ai_search.retrieval.query.embed_image", new_callable=AsyncMock)
    @patch("ai_search.retrieval.query.embed_text", new_callable=AsyncMock)
    @patch("ai_search.retrieval.query.get_openai_client")
    @patch("ai_search.retrieval.query.load_config")
    async def test_image_url_calls_embed_image(self, ...): ...

    # Test 3: Verify text-only queries call embed_text_for_image_search()
    @pytest.mark.asyncio()
    @patch("ai_search.retrieval.query.embed_text_for_image_search", new_callable=AsyncMock)
    @patch("ai_search.retrieval.query.embed_text", new_callable=AsyncMock)
    @patch("ai_search.retrieval.query.get_openai_client")
    @patch("ai_search.retrieval.query.load_config")
    async def test_text_only_calls_embed_text_for_image_search(self, ...): ...
```

### Fixtures needed

* No custom fixtures required — all mocks are via `@patch()` decorators
* `AppConfig()` is instantiated inline with defaults (same pattern as `test_search.py`)
* `mock_openai_client` from `conftest.py` could be used but inline setup is simpler and more explicit

### Mock return values

* `load_config()` → `AppConfig()` (default dimensions: semantic=3072, structural=1024, style=512, image=1024)
* `get_openai_client()` → `MagicMock` with `chat.completions.create()` returning mock with `choices[0].message.content = "test description"`
* `embed_text()` → side_effect function returning `[0.1] * dimensions` based on the `dimensions` kwarg
* `embed_image()` → `[0.4] * 1024`
* `embed_text_for_image_search()` → `[0.5] * 1024`

## Edge cases and discoveries

1. **`embed_text` is called 3 times** with different `dimensions` kwargs (3072, 1024, 512). A simple `return_value` won't suffice if dimension-specific assertions are needed. Use `side_effect` or a generic return like `[0.1] * 3072` (largest), or use `AsyncMock(return_value=[0.1] * 3072)` if tests only check keys, not vector lengths.

2. **LLM client is synchronous** despite `generate_query_vectors` being async. The `client.chat.completions.create()` call is not awaited—it's a regular synchronous call. The mock must be a regular `MagicMock`, not `AsyncMock`.

3. **The `embed_text` calls use keyword argument `dimensions`**: `embed_text(query_text, dimensions=dims.semantic)`. Side effect logic can use this to return appropriately sized vectors.

4. **Test 2 should also verify `embed_text_for_image_search` is NOT called**, and Test 3 should verify `embed_image` is NOT called, to confirm mutual exclusivity of the branches.

5. **All four dict keys should be asserted** in Test 1: `semantic_vector`, `structural_vector`, `style_vector`, `image_vector`.
