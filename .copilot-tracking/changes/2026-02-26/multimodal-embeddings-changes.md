<!-- markdownlint-disable-file -->
# Release Changes: Multimodal Image Embeddings

**Related Plan**: multimodal-embeddings-plan.instructions.md
**Implementation Date**: 2026-02-26

## Summary

Added Azure Computer Vision 4.0 (Florence) image embedding support to the AI Search pipeline. Documents can now include image vectors alongside text vectors, enabling cross-modal search where text queries match against image content and image queries match against text content. A text-boost scoring profile prioritizes generation prompts and tags in hybrid search results. Search weights were rebalanced to accommodate the new image channel.

## Changes

### Added

* `src/ai_search/embeddings/image.py` — Florence image embedding module with `embed_image()` (URL or bytes) and `embed_text_for_image_search()` functions, both producing 1024-dim vectors via the Computer Vision 4.0 REST API
* `tests/test_embeddings/__init__.py` — Package init for test embeddings module
* `tests/test_embeddings/test_image.py` — 5 unit tests covering URL embedding, bytes embedding, no-input error, API error handling, and text-for-image-search

### Modified

* `src/ai_search/config.py` — Added `AzureComputerVisionSecrets` class, `load_cv_secrets()` factory, `image_embedding_model` to `ModelsConfig`, `image: int = 1024` to `VectorDimensionsConfig`, `image_weight: float = 0.2` to `SearchWeightsConfig` (weights rebalanced: 0.4/0.15/0.15/0.2/0.1)
* `src/ai_search/clients.py` — Added `httpx` import, `load_cv_secrets` import, and `get_cv_client()` factory returning `httpx.AsyncClient` with API key header and 30s timeout
* `src/ai_search/embeddings/pipeline.py` — Extended `generate_all_vectors()` with optional `image_url`/`image_bytes` params; Florence embedding runs in parallel via `asyncio.gather()`, returning `image_vector` when image input provided
* `src/ai_search/models.py` — Added `image_vector: list[float] = Field(default_factory=list)` to `ImageVectors` and `SearchDocument`
* `src/ai_search/indexing/schema.py` — Added `image_vector` HNSW cosine field (1024 dims), `ScoringProfile`/`TextWeights` imports, text-boost scoring profile (generation_prompt 3.0x, tags 2.0x) set as default
* `src/ai_search/retrieval/query.py` — Added `query_image_url` param and Florence embedding calls for image-to-image and text-to-image cross-modal search vectors
* `src/ai_search/retrieval/search.py` — Added `semantic_vector` and `image_vector` to `SELECT_FIELDS`; added image vector query block with `image_weight * 10` weighting
* `config.yaml` — Added `image_embedding_model: azure-cv-florence`, `image: 1024` dimension, `image_weight: 0.2`; rebalanced existing weights
* `.env.example` — Added `AZURE_CV_ENDPOINT` and `AZURE_CV_API_KEY` placeholder entries
* `README.md` — Added Image Embeddings section (architecture, alternative models), Search Tier section (S2 Standard), Azure Resource Requirements section
* `tests/conftest.py` — Updated `sample_vectors` fixture with `image_vector` (1024 dims), `sample_config` fixture with `image_weight` and `image` dimension
* `tests/test_embeddings/test_pipeline.py` — Added `test_with_image_url` test; updated `test_parallel_execution` to assert `image_vector == []`
* `tests/test_retrieval/test_search.py` — Added `test_includes_image_vector_query` verifying 4 vector queries and correct field targeting
* `tests/test_config.py` — Fixed `test_search_weights_sum_to_one` to include `image_weight` in the sum
* `tests/test_indexing/test_schema.py` — Updated `test_field_count` from 27 to 28 fields (4 primary vectors)

### Removed

* None

## Additional or Deviating Changes

* `httpx>=0.27` was already present in `pyproject.toml` — Step 5.1 required no modification
  * Verified via `grep httpx pyproject.toml`
* Two existing tests required fixes after implementation (not anticipated in the plan):
  * `test_search_weights_sum_to_one` — did not account for new `image_weight` in sum validation
  * `test_field_count` — expected 27 fields, needed update to 28
  * Both fixed as part of Phase 7 validation

## Release Summary

**Total files affected**: 17 (2 created, 15 modified, 0 removed)

**Files created**:

* `src/ai_search/embeddings/image.py` — Florence image embedding functions
* `tests/test_embeddings/test_image.py` — Image embedding unit tests

**Files modified**:

* `src/ai_search/config.py` — CV secrets, image config extensions
* `src/ai_search/clients.py` — Florence HTTP client factory
* `src/ai_search/embeddings/pipeline.py` — Image embedding in parallel gather
* `src/ai_search/models.py` — image_vector field on models
* `src/ai_search/indexing/schema.py` — image_vector index field, scoring profile
* `src/ai_search/retrieval/query.py` — Image query vector generation
* `src/ai_search/retrieval/search.py` — Image vector in hybrid search
* `config.yaml` — Image embedding configuration
* `.env.example` — Florence credential placeholders
* `README.md` — Documentation updates
* `tests/conftest.py` — Shared fixture updates
* `tests/test_embeddings/test_pipeline.py` — Pipeline image test
* `tests/test_retrieval/test_search.py` — Search image vector test
* `tests/test_config.py` — Weights sum fix
* `tests/test_indexing/test_schema.py` — Field count fix

**Dependency changes**: None (httpx already present)

**Infrastructure changes**: Requires Azure Computer Vision resource with Florence model deployed. Index requires S2 Standard tier for 4 vector fields.

**Deployment notes**: Set `AZURE_CV_ENDPOINT` and `AZURE_CV_API_KEY` environment variables before running. Existing indexes must be recreated to add the `image_vector` field and scoring profile.

**Validation**: ruff ✅, mypy ✅ (33 source files), pytest ✅ (57 passed, 3 deselected)

**Test coverage**: 7 new tests added (5 image embedding, 1 pipeline, 1 search). Total: 57 passing.
