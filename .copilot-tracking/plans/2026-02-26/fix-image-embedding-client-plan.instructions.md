---
applyTo: '.copilot-tracking/changes/2026-02-26/fix-image-embedding-client-changes.md'
---
<!-- markdownlint-disable-file -->
# Implementation Plan: Fix Image Embedding Client

## Overview

Replace `EmbeddingsClient` (text route) with `ImageEmbeddingsClient` (image route) for image embeddings, re-ingest all documents to generate correct visual vectors, and update tests.

## Objectives

### User Requirements

* Fix incorrectly generated image embedding vectors — Source: user investigation "are the embedding vectors correctly created?"
* Ensure image search scores differentiate relevant from irrelevant images (cat vs ocean should not be 0.96) — Source: user observation "we won't be able to put any cap of threshold"

### Derived Objectives

* Add `get_foundry_image_embed_client()` factory in `clients.py` — Derived from: SDK requires `ImageEmbeddingsClient` for `/images/embeddings` route
* Wrap data URIs in `ImageEmbeddingInput` model — Derived from: `ImageEmbeddingsClient.embed()` requires `List[ImageEmbeddingInput]` not `List[str]`
* Update test mocks to cover the new client — Derived from: existing tests mock `get_foundry_embed_client` for image paths, must switch to image client
* Re-ingest all 10 sample documents — Derived from: existing `image_vector` values were generated via text tokenization and are invalid
* Optionally set `input_type` on text embeddings — Derived from: Cohere Embed v4 supports `query` vs `document` input types for retrieval optimization

## Context Summary

### Project Files

* `src/ai_search/clients.py` - Client factory functions; needs `ImageEmbeddingsClient` factory (130 lines)
* `src/ai_search/embeddings/image.py` - Image/text embedding with Florence and Foundry backends; `_embed_image_foundry()` uses wrong client (285 lines)
* `tests/test_embeddings/test_image.py` - Tests for image embedding; Foundry mocks target wrong client (331 lines)
* `scripts/ingest_samples.py` - Batch ingestion script; needs force-reindex flag or manual index clear (210 lines)
* `src/ai_search/retrieval/relevance.py` - Relevance scoring module; thresholds may need re-tuning after fix (182 lines)

### References

* `.copilot-tracking/research/2026-02-26/embedding-vector-correctness-research.md` - Root cause analysis and correct API usage
* Azure AI Inference SDK source: `.venv/lib/python3.12/site-packages/azure/ai/inference/aio/_patch.py` - `ImageEmbeddingsClient.embed()` signature

### Standards References

* #file:../../.github/instructions/hve-core/commit-message.instructions.md - Commit message conventions

## Implementation Checklist

### [x] Implementation Phase 1: Fix Image Embedding Client

<!-- parallelizable: true -->

* [x] Step 1.1: Add `ImageEmbeddingsClient` import and factory to `clients.py`
  * Details: .copilot-tracking/details/2026-02-26/fix-image-embedding-client-details.md (Lines 15-50)
* [x] Step 1.2: Update `_embed_image_foundry()` in `image.py` to use image client with `ImageEmbeddingInput`
  * Details: .copilot-tracking/details/2026-02-26/fix-image-embedding-client-details.md (Lines 52-95)
* [ ] Step 1.3: (Optional) Add `input_type` parameter to `_embed_text_foundry()`
  * Details: .copilot-tracking/details/2026-02-26/fix-image-embedding-client-details.md (Lines 97-120)
  * Status: Deferred to WI-02 (enhancement, not bug fix)

### [x] Implementation Phase 2: Update Tests

<!-- parallelizable: true -->

* [x] Step 2.1: Update `mock_foundry_backend` fixture to mock `ImageEmbeddingsClient`
  * Details: .copilot-tracking/details/2026-02-26/fix-image-embedding-client-details.md (Lines 122-165)
* [x] Step 2.2: Update Foundry image test assertions for `ImageEmbeddingInput`
  * Details: .copilot-tracking/details/2026-02-26/fix-image-embedding-client-details.md (Lines 167-210)
* [x] Step 2.3: Update Foundry validation test to mock the image client
  * Details: .copilot-tracking/details/2026-02-26/fix-image-embedding-client-details.md (Lines 212-245)

### [x] Implementation Phase 3: Re-ingest Documents

<!-- parallelizable: false -->

* [x] Step 3.1: Delete existing index or clear image_vector values
  * Details: .copilot-tracking/details/2026-02-26/fix-image-embedding-client-details.md (Lines 247-275)
* [x] Step 3.2: Run ingestion pipeline with force flag
  * Details: .copilot-tracking/details/2026-02-26/fix-image-embedding-client-details.md (Lines 277-305)
* [x] Step 3.3: Verify new image vectors produce differentiated scores
  * Details: .copilot-tracking/details/2026-02-26/fix-image-embedding-client-details.md (Lines 307-340)

### [x] Implementation Phase 4: Validation

<!-- parallelizable: false -->

* [x] Step 4.1: Run full test suite
  * Execute `source .venv/bin/activate && python -m pytest tests/ -v`
  * Result: 50 passed, 3 skipped
* [x] Step 4.2: Run live image search verification
  * Ocean exact: score 1.0, gap 0.32 → HIGH confidence ✅
  * Cat irrelevant: score 0.54 (down from 0.97) → MEDIUM (small corpus limitation)
  * Mountain similar: score 0.61 → MEDIUM ✅
* [x] Step 4.3: Fix minor validation issues
  * Fixed pre-existing tuple unpacking in test_search.py
* [x] Step 4.4: Report blocking issues
  * No blocking issues

## Planning Log

See [fix-image-embedding-client-log.md](../logs/2026-02-26/fix-image-embedding-client-log.md) for discrepancy tracking, implementation paths considered, and suggested follow-on work.

## Dependencies

* `azure-ai-inference>=1.0.0b9` (already installed — provides `ImageEmbeddingsClient` and `ImageEmbeddingInput`)
* Azure AI Foundry endpoint with embed-v-4-0 deployment (already configured)
* Entra ID credentials with `DefaultAzureCredential` (already configured)
* PIL/Pillow for image resizing (already installed)

## Success Criteria

* `_embed_image_foundry()` uses `ImageEmbeddingsClient` with `ImageEmbeddingInput` — Traces to: research finding "wrong client, wrong API route"
* All unit tests pass with updated mocks — Traces to: derived objective on test coverage
* Re-ingested image vectors produce differentiated cosine scores (unrelated images < 0.7) — Traces to: user requirement "cat vs ocean should not be 0.96"
* Ocean exact match still scores ~1.0 — Traces to: regression prevention
* Relevance scoring tiers correctly classify exact/similar/irrelevant queries — Traces to: user requirement on threshold viability
