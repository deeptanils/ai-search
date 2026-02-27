<!-- markdownlint-disable-file -->
# RPI Validation: Phases 4-7

**Plan**: multimodal-embeddings-plan.instructions.md
**Changes Log**: multimodal-embeddings-changes.md
**Details**: multimodal-embeddings-details.md (Lines 510-745)
**Validation Date**: 2026-02-26
**Validator**: RPI Automated Validation

## Phase 4: Retrieval Layer

### Step 4.1: PASSED

**Plan**: Update `retrieval/query.py` to support image query vectors.

**Verification**:

* Import `embed_image` and `embed_text_for_image_search` from `ai_search.embeddings.image` — **Present** (Line 10)
* `generate_query_vectors()` signature updated with `query_image_url: str | None = None` — **Present** (Lines 27-29)
* Conditional image embedding logic: `embed_image(image_url=...)` when URL provided, `embed_text_for_image_search(query_text)` otherwise — **Present** (Lines 75-79)
* Return dict includes `"image_vector"` key — **Present** (Line 84)
* `generate_query_vectors_sync()` updated with `query_image_url` param — **Present** (Lines 89-93)
* All success criteria met: text query generates 4 vectors, image query generates 4 vectors, `image_vector` key always present

**Evidence**: `src/ai_search/retrieval/query.py` (Lines 1-93) matches plan specification exactly.

### Step 4.2: PASSED

**Plan**: Update `retrieval/search.py` to include `image_vector` in hybrid search.

**Verification**:

* `semantic_vector` added to `SELECT_FIELDS` — **Present** (Line 27)
* `image_vector` added to `SELECT_FIELDS` — **Present** (Line 28)
* Image vector query block with `VectorizedQuery`, `fields="image_vector"`, `weight=weights.image_weight * 10` — **Present** (Lines 85-92)
* Conditional guard `if "image_vector" in query_vectors` — **Present** (Line 85)
* Weight follows ×10 pattern consistent with other vectors — **Confirmed**

**Evidence**: `src/ai_search/retrieval/search.py` (Lines 14-92) matches plan specification exactly. IV-014 fix also applied (semantic_vector in SELECT_FIELDS).

## Phase 5: Dependencies & Documentation

### Step 5.1: PASSED

**Plan**: Verify `httpx>=0.27` is present in `pyproject.toml`.

**Verification**:

* `httpx>=0.27` is in the `dependencies` list — **Present** (Line 19 of `pyproject.toml`)
* No modification was needed; dependency was already present
* Changes log correctly documents this as a no-op: "httpx>=0.27 was already present in pyproject.toml"

**Evidence**: `pyproject.toml` Line 19: `"httpx>=0.27",`

### Step 5.2: PASSED

**Plan**: Update `README.md` with image embedding setup and S2 tier guidance.

**Verification**:

* Azure Computer Vision resource provisioning — **Present** (Azure Resource Requirements section)
* `.env` configuration for `AZURE_CV_ENDPOINT` and `AZURE_CV_API_KEY` — **Present** (code block with credential examples)
* How image embeddings work alongside text embeddings — **Present** (Image Embeddings section with architecture description)
* Cohere Embed v4 documented as alternative — **Present** ("If a strict Azure AI Foundry-only constraint applies, Cohere Embed v4...")
* S2 Standard tier recommendation with storage calculation — **Present** (Search Tier section: "~37 GB at 10M documents")
* Foundry-only note — **Present** ("This is a separate Azure resource, not part of the AI Foundry model catalog")

**Minor gap**: README does not explicitly reference DD-01 (Foundry deviation document in the planning log). It acknowledges the separate-resource nature but omits the cross-reference. See RPI-006.

**Evidence**: `README.md` (Lines 1-113) covers all 6 required content items.

## Phase 6: Tests

### Step 6.1: PASSED

**Plan**: Add 5 unit tests for `embeddings/image.py`.

**Verification**:

| # | Test Name | Plan Spec | Status |
|---|-----------|-----------|--------|
| 1 | `test_embed_image_url` | Mock Florence API, verify 1024-dim vector from URL | **Present** — asserts `len(result) == 1024`, verifies `vectorizeImage` in URL, checks JSON payload |
| 2 | `test_embed_image_bytes` | Mock Florence API, verify 1024-dim vector from bytes | **Present** — asserts `len(result) == 1024`, verifies `content` kwarg |
| 3 | `test_embed_image_no_input_raises` | Verify `ValueError` when neither URL nor bytes | **Present** — `pytest.raises(ValueError, match="Either image_url or image_bytes must be provided")` |
| 4 | `test_embed_text_for_image_search` | Mock `vectorizeText`, verify 1024-dim vector | **Present** — asserts `len(result) == 1024`, verifies `vectorizeText` in URL, checks JSON payload |
| 5 | `test_embed_image_api_error` | Mock 500 response, verify `HTTPStatusError` | **Present** — mocks `raise_for_status` side effect, asserts `pytest.raises(httpx.HTTPStatusError)` |

* Mock pattern uses `monkeypatch.setattr` for `get_cv_client` and `load_cv_secrets` — **Matches** plan specification
* All 5 tests pass without Azure credentials — **Confirmed** per changes log (57 passed)

**Evidence**: `tests/test_embeddings/test_image.py` (Lines 1-99) — all 5 tests present and correctly structured.

### Step 6.2: PASSED

**Plan**: Update existing embedding pipeline test for image vector.

**Verification**:

* `test_parallel_execution` updated to assert `result.image_vector == []` when no image input — **Present** (Line 43)
* `test_with_image_url` added to verify `image_vector` populated when `image_url` provided — **Present** (Lines 48-85)
  * Verifies `len(result.image_vector) == 1024` — **Present** (Line 82)
  * Verifies `mock_embed_image.assert_called_once_with(image_url=..., image_bytes=None)` — **Present** (Lines 83-86)

**Evidence**: `tests/test_embeddings/test_pipeline.py` (Lines 1-86) — both backward-compat and new image tests present.

### Step 6.3: PARTIAL

**Plan**: Add unit tests for updated search and query modules.

**Search test verification**:

* `test_includes_image_vector_query` — **Present** in `tests/test_retrieval/test_search.py` (Lines 50-73)
  * Verifies 4 vector queries when image_vector present — **Present** (`assert len(...) == 4`)
  * Verifies correct field targeting `image_query.fields == "image_vector"` — **Present**
  * Weight verification (`image_weight * 10`) — **Not explicitly tested**. The test checks field name but does not assert the weight value. See RPI-005.

**Query test verification**:

* Plan specifies creating `tests/test_retrieval/test_query.py` as a **NEW file** — **FILE NOT FOUND** ❌
* Changes log does not mention `test_query.py` — **Confirmed missing**
* Specified tests not implemented:
  * Mock LLM + embedding + Florence clients — **Missing**
  * Verify `generate_query_vectors()` returns dict with `image_vector` key — **Missing**
  * Verify image-only query calls `embed_image()` instead of `embed_text_for_image_search()` — **Missing**

**Evidence**: `tests/test_retrieval/test_query.py` does not exist (file_search returned no results). See RPI-004.

## Phase 7: Validation

### Step 7.1: PASSED

**Plan**: Run full project validation (ruff, mypy, pytest).

**Verification** (from changes log):

* `uv run ruff check src/ tests/` — ✅ (zero errors)
* `uv run mypy src/` — ✅ (33 source files, zero errors)
* `uv run pytest tests/ -m "not integration"` — ✅ (57 passed, 3 deselected)

**Evidence**: Changes log "Validation" section confirms all three tools passed.

### Step 7.2: PASSED

**Plan**: Fix minor validation issues.

**Verification**:

* `tests/test_config.py` — `test_search_weights_sum_to_one` updated to include `config.search.image_weight` in the sum — **Present** (Lines 42-48)
  * Sum: `semantic_weight (0.4) + structural_weight (0.15) + style_weight (0.15) + image_weight (0.2) + keyword_weight (0.1) = 1.0` — **Correct**
* `tests/test_indexing/test_schema.py` — `test_field_count` updated from 27 to 28 — **Present** (Line 19)
  * Comment: "15 primitive fields + 4 primary vectors + 9 character vectors (3 slots × 3) = 28" — **Arithmetically correct**
* Both fixes are sensible responses to the new `image_vector` field and `image_weight` config additions

**Evidence**: `tests/test_config.py` (Lines 39-48), `tests/test_indexing/test_schema.py` (Lines 14-19).

### Step 7.3: PASSED

**Plan**: Report blocking issues.

**Verification**:

* No blocking issues reported in the changes log — **Correct**
* All validation tools passed clean — **Confirmed**
* Changes log documents two non-blocking deviations (pre-existing httpx dep, two test fixes) — **Properly documented**

## Findings Summary

| Severity | Count |
|----------|-------|
| Critical | 0     |
| Major    | 1     |
| Minor    | 2     |

## Detailed Findings

### RPI-004 — Major: Missing query vector generation test file

**Severity**: Major
**Phase**: 6 (Tests), Step 6.3
**Description**: The plan specifies creating `tests/test_retrieval/test_query.py` as a NEW file with tests for `generate_query_vectors()`. This file was not created and is not mentioned in the changes log.

**Plan specification**:

> Query test:
> * Mock LLM + embedding + Florence clients
> * Verify `generate_query_vectors()` returns dict with `image_vector` key
> * Verify image-only query calls `embed_image()` instead of `embed_text_for_image_search()`

**Evidence**: `file_search` for `tests/test_retrieval/test_query.py` returned no results. The changes log "Files created" section lists only 2 files (`image.py`, `test_image.py`), neither of which is the query test. The "Files modified" section also does not mention `test_query.py`.

**Impact**: The `generate_query_vectors()` function's new image embedding logic (cross-modal text-to-image and direct image-to-image branching) has no dedicated unit test coverage. This is a gap in verifying the core Phase 4.1 implementation.

**Recommendation**: Create `tests/test_retrieval/test_query.py` with the tests specified in the plan. At minimum, verify that `generate_query_vectors()` returns a dict containing `image_vector`, and that the image URL path calls `embed_image()` while the text-only path calls `embed_text_for_image_search()`.

### RPI-005 — Minor: Image vector weight not explicitly verified in search test

**Severity**: Minor
**Phase**: 6 (Tests), Step 6.3
**Description**: `test_includes_image_vector_query` verifies that 4 vector queries are created and that the image query targets the correct field, but does not assert that the weight equals `image_weight * 10`.

**Plan specification**:

> * Verify `image_weight * 10` is used as the weight

**Evidence**: `tests/test_retrieval/test_search.py` Lines 69-73:

```python
assert len(call_kwargs.kwargs["vector_queries"]) == 4
image_query = call_kwargs.kwargs["vector_queries"][3]
assert image_query.fields == "image_vector"
```

No assertion on `image_query.weight`.

**Impact**: A regression changing the weight calculation would not be caught by this test.

**Recommendation**: Add `assert image_query.weight == pytest.approx(0.2 * 10)` (or equivalent using `AppConfig().search.image_weight * 10`) to the test.

### RPI-006 — Minor: README omits DD-01 cross-reference for Foundry deviation

**Severity**: Minor
**Phase**: 5 (Dependencies & Documentation), Step 5.2
**Description**: The plan specifies documenting a "Foundry-only waiver note: Florence is a separate Azure Cognitive Services resource (see DD-01 in planning log)." The README acknowledges Florence as a separate resource but does not reference DD-01 or the planning log.

**Evidence**: `README.md` Line 46: "This is a separate Azure resource, not part of the AI Foundry model catalog." — no mention of DD-01 or a link to the planning log.

**Plan specification**:

> 6. Foundry-only waiver note: Florence is a separate Azure Cognitive Services resource (see DD-01 in planning log)

**Impact**: Developers reviewing the README may not be aware that the Foundry-only deviation was explicitly discussed and approved. The deviation is documented in the planning log but not discoverable from the README.

**Recommendation**: Add a note such as: "See DD-01 in the planning log for the Foundry-only waiver rationale." or link to the planning log file.

## Cross-Reference: Research Document

Validated against `.copilot-tracking/research/2026-02-26/model-strategy-index-config-research.md`:

* **Florence REST API pattern** (Q2): Implementation uses `httpx` for `vectorizeImage`/`vectorizeText` endpoints — **Matches** research recommendation (Option A: REST API recommended)
* **1024-dim fixed output**: `image_vector` field uses 1024 dims throughout config, schema, and tests — **Matches** research finding
* **Shared embedding space**: `embed_text_for_image_search()` enables cross-modal queries in Florence space — **Matches** Q2 "Shared embedding space" characteristic
* **Option D architecture**: Implementation follows the recommended hybrid architecture (text-embedding-3-large + Florence) — **Matches** research recommendation
* **Config changes**: All recommended config additions (image_embedding_model, image dimension, CV secrets, image_weight) implemented — **Matches** research "Config Changes Needed" section
* **`httpx` as client**: Research recommends REST via `httpx` over `azure-ai-vision-imageanalysis` SDK — **Matches** implementation choice
* **Cohere documented as alternative**: README mentions Cohere Embed v4 — **Matches** research Q3 findings

No conflicts found between research document and implementation.

## Phase Summary

| Phase | Status  | Notes |
|-------|---------|-------|
| 4     | Passed  | Both steps fully implemented per spec |
| 5     | Passed  | httpx verified, README covers all required content (minor DD-01 gap) |
| 6     | Partial | 2 of 3 steps passed; `test_query.py` missing (RPI-004) |
| 7     | Passed  | All validation tools passed, test fixes correctly applied |
