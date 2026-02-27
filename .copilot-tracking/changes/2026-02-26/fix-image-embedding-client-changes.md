<!-- markdownlint-disable-file -->
# Release Changes: Fix Image Embedding Client

**Related Plan**: fix-image-embedding-client-plan.instructions.md
**Implementation Date**: 2026-02-26

## Summary

Replaced `EmbeddingsClient` (text route `POST /embeddings`) with `ImageEmbeddingsClient` (image route `POST /images/embeddings`) for image embeddings. This fixes the root cause of uniformly high cosine similarity scores (0.95+) for all image searches — images were being tokenized as base64 text instead of processed as visual content. All 10 sample documents re-indexed with correct image vectors.

## Changes

### Added

* src/ai_search/clients.py - Added `get_foundry_image_embed_client()` factory returning `ImageEmbeddingsClient` with Entra ID auth, following existing `get_foundry_embed_client()` pattern

### Modified

* src/ai_search/clients.py - Added `ImageEmbeddingsClient` to `azure.ai.inference.aio` import
* src/ai_search/embeddings/image.py - Added `ImageEmbeddingInput` import from `azure.ai.inference.models` and `get_foundry_image_embed_client` from clients
* src/ai_search/embeddings/image.py - Rewrote `_embed_image_foundry()` to use `ImageEmbeddingsClient` with `ImageEmbeddingInput(image=data_uri)` wrapper instead of passing data URI as plain string to `EmbeddingsClient`
* tests/test_embeddings/test_image.py - Updated `mock_foundry_backend` fixture to mock both `get_foundry_image_embed_client` (image route) and `get_foundry_embed_client` (text route) with descriptive keys `image_embed_client` / `text_embed_client`
* tests/test_embeddings/test_image.py - Updated `TestFoundryEmbedImage` assertions to check `ImageEmbeddingInput.image` attribute instead of raw string
* tests/test_embeddings/test_image.py - Updated `TestFoundryEmbedText` to use `text_embed_client` key from fixture
* tests/test_embeddings/test_image.py - Updated `TestFoundryValidation` to mock `get_foundry_image_embed_client` instead of `get_foundry_embed_client`
* tests/test_retrieval/test_search.py - Fixed pre-existing tuple unpacking for `execute_hybrid_search()` return type
* scripts/ingest_samples.py - Added `--force` flag to re-index documents that are already in the index, updated `run()` signature and skip-logic conditional

### Removed

* (none)

## Additional or Deviating Changes

* Fixed pre-existing test failure in `test_search.py::test_builds_vector_queries` — test was not updated when `execute_hybrid_search()` return type changed to `tuple[list, RelevanceResult | None]` in a previous session
  * Reason: discovered during Phase 4 validation; fix was straightforward (tuple unpacking)

* Step 1.3 (optional `input_type` for text embeddings) was not implemented
  * Reason: deferred to follow-on work WI-02 as documented in planning log; enhancement, not bug fix

* Cat image still classifies as MEDIUM confidence (0.54 score) due to 10-document corpus limitations
  * Reason: documented as DD-01 in planning log; relative metrics (z-score, gap ratio, spread) still produce MEDIUM when all scores are compressed in a small corpus; threshold re-tuning deferred to WI-01 after larger corpus testing

## Release Summary

Total files affected: 5 (4 modified, 0 created, 0 removed within src/tests/scripts)

Files modified:
* `src/ai_search/clients.py` — Added `ImageEmbeddingsClient` import and `get_foundry_image_embed_client()` factory
* `src/ai_search/embeddings/image.py` — Switched `_embed_image_foundry()` to `ImageEmbeddingsClient` + `ImageEmbeddingInput`
* `tests/test_embeddings/test_image.py` — Updated fixture and all Foundry test assertions for new client
* `tests/test_retrieval/test_search.py` — Fixed tuple unpacking for `execute_hybrid_search()` return
* `scripts/ingest_samples.py` — Added `--force` re-indexing flag

Dependency changes: None (all required classes were already available in `azure-ai-inference>=1.0.0b9`)

Deployment notes:
* All 10 documents in `candidate-index` have been re-indexed with corrected image vectors
* Image search scores now differentiate: exact match=1.0, similar=0.6-0.7, unrelated=0.53
* The `--force` flag should be used when re-indexing after embedding model changes
* Score range changed from 0.94-1.0 (broken) to 0.53-1.0 (correct) — any threshold logic should be reviewed
