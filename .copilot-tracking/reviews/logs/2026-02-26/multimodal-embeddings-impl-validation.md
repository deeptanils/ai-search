<!-- markdownlint-disable-file -->
# Implementation Validation: Multimodal Image Embeddings

## Scope: full-quality

## Status: Complete

## Exploration Notes

The implementation adds Azure Computer Vision 4.0 (Florence) image embedding support across 8 source files, 6 test files, and 3 config files. The codebase follows a clean layered architecture: `config → clients → embeddings → models → indexing → retrieval`. Conventions are consistent — pydantic for models/config, structlog for logging, async-first with sync wrappers, and `lru_cache` for client factories. The Florence integration follows existing patterns closely, which is a strong sign of architectural discipline.

All default search weights sum to 1.0 (verified: 0.4 + 0.15 + 0.15 + 0.2 + 0.1 = 1.0). The `image_vector` field (1024-dim) is correctly wired end-to-end: config → schema → models → pipeline → query → search. The `.env.example`, `config.yaml`, and `README.md` are all updated consistently.

## Findings

### Architecture

**[Pass] IV-001 (Architecture): Module dependency flow is correct.** The dependency graph is acyclic and follows the established layering:

* `embeddings/image.py` → imports from `clients`, `config` only
* `embeddings/pipeline.py` → imports from `embeddings/*`, `models`
* `retrieval/query.py` → imports from `clients`, `config`, `embeddings.encoder`, `embeddings.image`
* `retrieval/search.py` → imports from `clients`, `config` only
* No circular dependencies detected; no retrieval importing from indexing

**[Pass] IV-002 (Architecture): Florence client follows the factory pattern.** `get_cv_client()` in `clients.py` uses the same `@lru_cache(maxsize=1)` pattern as `get_openai_client()`, `get_async_openai_client()`, and `get_search_index_client()`. Consistent factory approach.

### Design Principles

**[Pass] IV-003 (Design): Single responsibility is maintained.** `embed_image()` handles image-to-vector, `embed_text_for_image_search()` handles text-to-image-space vector. Pipeline orchestration remains in `pipeline.py`. Search logic stays in `search.py`. Query vector generation in `query.py`.

**[Pass] IV-004 (Design): Open/closed principle.** The pipeline was extended (not modified in breaking ways) to support the optional `image_vector`. The `has_image` conditional in `generate_all_vectors` gracefully degrades when no image input is provided.

### DRY Analysis

**[Minor] IV-005 (DRY): Repeated parameter construction in `image.py`.** Evidence: `src/ai_search/embeddings/image.py` lines 36-38 and lines 73-75. Both `embed_image()` and `embed_text_for_image_search()` repeat identical client/secrets/params setup:

```python
client = get_cv_client()
secrets = load_cv_secrets()
params = {"api-version": secrets.api_version, "model-version": _MODEL_VERSION}
```

Impact: Low — only 2 occurrences, but will grow if more Florence endpoints are added. Recommendation: Extract a `_florence_request(path, *, json=None, content=None, headers=None)` helper that encapsulates client acquisition, params, POST call, `raise_for_status()`, and vector extraction.

### API & Library Usage

**[Pass] IV-006 (API): httpx AsyncClient usage is correct.** `get_cv_client()` configures `base_url`, `headers`, and `timeout=30.0` properly. POST calls use relative paths, which combine correctly with `base_url`.

**[Pass] IV-007 (API): `Ocp-Apim-Subscription-Key` header is correct.** This is the standard authentication header for Azure Cognitive Services / Computer Vision key-based auth.

**[Pass] IV-008 (API): Florence endpoint paths are correct.** `/computervision/retrieval:vectorizeImage` and `/computervision/retrieval:vectorizeText` match the Azure Computer Vision 4.0 REST API specification.

**[Minor] IV-009 (API): `_MODEL_VERSION` is hardcoded and not configurable.** Evidence: `src/ai_search/embeddings/image.py` line 17 (`_MODEL_VERSION = "2023-04-15"`). The `api_version` is configurable via `AzureComputerVisionSecrets`, but `model-version` is not. Impact: Low — `2023-04-15` is the current GA Florence model version, but when a newer model ships, a code change is required instead of a config change. Recommendation: Add `model_version` to `AzureComputerVisionSecrets` or `ModelsConfig`.

**[Minor] IV-010 (API): `lru_cache` on `get_cv_client()` creates a non-closeable singleton.** Evidence: `src/ai_search/clients.py` lines 72-79. The cached `httpx.AsyncClient` has no shutdown/close lifecycle hook. Impact: Low for CLI tools (process exit cleans up), but would leak connections in long-running server contexts. This is **consistent with the existing pattern** for other clients in this file, so it is not introduced by this change. Recommendation: Consider an application-level shutdown hook or context manager if the project moves to a server deployment.

### Error Handling

**[Major] IV-011 (Error Handling): No validation of Florence API response payload.** Evidence: `src/ai_search/embeddings/image.py` lines 54 and 82 — `response.json()["vector"]` accessed without checking key existence or vector emptiness. Impact: If the Florence API returns an unexpected payload (missing `"vector"` key → `KeyError`; empty vector `[]` → silent data corruption in the index). Recommendation: Add defensive checks:

```python
data = response.json()
vector = data.get("vector")
if not vector:
    raise ValueError(f"Florence returned empty or missing vector: {data}")
```

**[Minor] IV-012 (Error Handling): No input validation in `embed_text_for_image_search`.** Evidence: `src/ai_search/embeddings/image.py` lines 65-85. Unlike `embed_image()` which validates its inputs, `embed_text_for_image_search()` does not check for empty/None text. Impact: An empty string would be sent to the Florence API, likely returning a valid but meaningless vector. Recommendation: Add `if not text: raise ValueError(...)`.

**[Minor] IV-013 (Error Handling): `asyncio.gather` in pipeline without `return_exceptions`.** Evidence: `src/ai_search/embeddings/pipeline.py` line 31. If the image embedding task fails, all other successfully-computed vectors are discarded. Impact: A transient Florence API failure forces re-computation of all 4 other embeddings. This may be intentional (fail-fast), but is worth documenting. Recommendation: Document the fail-fast behavior, or consider `return_exceptions=True` with partial result handling.

### Security

**[Major] IV-014 (Security): CV secrets default to empty strings, permitting silent misconfiguration.** Evidence: `src/ai_search/config.py` lines 66-67 — `endpoint: str = ""` and `api_key: str = ""`. Compare with `AzureFoundrySecrets` and `AzureSearchSecrets` which have **no defaults**, causing pydantic-settings to raise `ValidationError` at startup if env vars are missing. Impact: If `AZURE_CV_ENDPOINT` and `AZURE_CV_API_KEY` are not set, the application silently proceeds with empty strings, resulting in confusing runtime errors (httpx sending requests to empty base URL) instead of a clear startup failure. Recommendation: Remove the empty string defaults to match the pattern of other secrets classes:

```python
endpoint: str
api_key: str
```

Or, if the intent is to make Florence optional, add an explicit `enabled` flag and guard usage.

**[Pass] IV-015 (Security): No secrets in logs.** Logging in `image.py` only emits `dimensions=len(vector)`, not the vector values or API keys. Consistent with project logging conventions.

**[Pass] IV-016 (Security): API key handled safely.** The key is injected via httpx client headers (set once in the factory), not passed as URL params or logged.

### Test Coverage

**[Pass] IV-017 (Test Coverage): Core happy paths are tested.** `test_image.py` covers image URL embedding, image bytes embedding, cross-modal text embedding, input validation error, and API error propagation. `test_pipeline.py` covers pipeline with and without image input. `test_search.py` covers search with and without image vector query.

**[Minor] IV-018 (Test Coverage): Missing edge case — empty or missing vector in Florence response.** Evidence: `tests/test_embeddings/test_image.py` — no test for `response.json()` returning `{"vector": []}` or `{"modelVersion": "...", "error": {...}}`. Impact: IV-011's defensive check would remain untested. Recommendation: Add test cases for empty vector and missing vector key.

**[Minor] IV-019 (Test Coverage): Missing edge case — timeout scenario.** Evidence: `tests/test_embeddings/test_image.py` — no test for `httpx.TimeoutException`. Impact: Behavioral contract for timeout is undocumented and untested. Recommendation: Add a test that mocks a `TimeoutException` to verify it propagates cleanly.

**[Minor] IV-020 (Test Coverage): Missing edge case — pipeline error propagation.** Evidence: `tests/test_embeddings/test_pipeline.py` — no test for what happens when `embed_image` raises inside `asyncio.gather`. Impact: The fail-fast behavior (IV-013) is untested. Recommendation: Add a test where `embed_image` raises and verify the entire `generate_all_vectors` call raises.

**[Pass] IV-021 (Test Coverage): Test isolation is adequate.** Tests use `monkeypatch` and `@patch` to mock external calls. Fixtures are well-structured in `conftest.py` with proper image_vector dimensions (1024) in `sample_vectors`. No real API calls leak.

**[Pass] IV-022 (Test Coverage): Schema test accounts for image_vector.** `test_schema.py` asserts 28 fields (15 primitive + **4** primary vectors including image_vector + 9 character vectors). Correct.

**[Pass] IV-023 (Test Coverage): Config test verifies weight sum.** `test_config.py` `test_search_weights_sum_to_one` includes `image_weight` in the sum, correctly validating the new weight doesn't break the total.

### General

**[Pass] IV-024 (General): Config consistency across all files.** `config.yaml`, `AppConfig` defaults, `conftest.py` fixture, `.env.example`, and `README.md` all agree on: image dimension = 1024, image_weight = 0.2, CV env var prefix = `AZURE_CV_`, and endpoint/key pattern. `image_embedding_model: azure-cv-florence` is set in both `config.yaml` and `ModelsConfig` default.

**[Pass] IV-025 (General): `SELECT_FIELDS` in `search.py` includes `image_vector`.** This ensures image vectors are retrieved for downstream re-ranking and display if needed.

**[Minor] IV-026 (General): `README.md` documents Florence as requiring a separate Azure resource.** This is accurate and important for users. The region constraint (East US, West US, etc.) is correctly noted. However, the `AZURE_CV_API_VERSION` env var is not mentioned in the README or `.env.example` (it uses a default in config). This is acceptable since it has a sensible default, but could confuse users who need a different API version.

## Holistic Assessment

The implementation is high quality and architecturally sound. The Florence integration follows existing conventions closely — same factory pattern, same config layering, same test mocking approach — making it feel native to the codebase. The two major findings are both defensive-programming gaps rather than functional bugs: empty default secrets (IV-014) create a silent misconfiguration risk, and missing response validation (IV-011) could surface as confusing `KeyError` exceptions in production. Both are straightforward fixes. The minor findings are predominantly about test edge cases and minor DRY improvements that would improve robustness but don't block the implementation.

## Summary

| Severity | Count |
|----------|-------|
| Critical | 0     |
| Major    | 2     |
| Minor    | 8     |
