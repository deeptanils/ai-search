---
applyTo: '.copilot-tracking/changes/2026-02-26/multimodal-embeddings-changes.md'
---
<!-- markdownlint-disable-file -->
# Implementation Plan: Multimodal Image Embeddings & Index Enhancements

## Overview

Add direct image-to-vector embedding via Azure Computer Vision 4.0 (Florence) alongside the existing text-embedding-3-large pipeline, enable text-boost scoring profiles in Azure AI Search, and update configuration for S2 tier planning.

## Objectives

### User Requirements

* Support direct image embeddings (image → vector) in addition to text embeddings — Source: user request ("Embeddings are currently text and Image")
* Recommend and configure models for Azure AI Foundry — Source: user request ("suggest me models I will enable this inside AI Foundry")
* Improve Azure AI Search index configuration — Source: user request ("AI search index you can suggest what should be best for this use case")

### Derived Objectives

* Add Azure Computer Vision 4.0 (Florence) as the image embedding provider — Derived from: research finding that Florence is the best-fit managed model for direct image-to-vector (1024-dim shared text-image space)
* Document Cohere Embed v4 as a fallback if strict Foundry-only constraint applies — Derived from: user selected "Both (Florence primary, Cohere fallback)"
* Add a text-boost scoring profile for `generation_prompt` and `tags` fields — Derived from: user selected scoring profiles for index optimization
* Add `image_vector` field (1024 dims) to Azure AI Search index schema — Derived from: Florence produces fixed 1024-dim vectors
* Add `image_weight` to search weights for hybrid retrieval — Derived from: new vector field requires weighted inclusion in RRF

## Context Summary

### Project Files

* `src/ai_search/config.py` (Lines 1-172) — Configuration management: `ModelsConfig`, `SearchWeightsConfig`, `VectorDimensionsConfig`, `AppConfig`
* `src/ai_search/clients.py` (Lines 1-68) — Client factories: `get_openai_client()`, `get_async_openai_client()`, `get_search_index_client()`, `get_search_client()`
* `src/ai_search/embeddings/pipeline.py` (Lines 1-31) — Embedding orchestrator: `generate_all_vectors()`
* `src/ai_search/embeddings/encoder.py` (Lines 1-58) — Text embedding: `embed_texts()`, `embed_text()`
* `src/ai_search/indexing/schema.py` (Lines 1-140) — Index schema definition: `build_index_schema()`
* `src/ai_search/retrieval/search.py` (Lines 1-96) — Hybrid search: `execute_hybrid_search()`, `SELECT_FIELDS`
* `src/ai_search/retrieval/query.py` (Lines 1-78) — Query embedding: `generate_query_vectors()`
* `src/ai_search/models.py` (Lines 1-172) — Pydantic models: `SearchDocument`, `ImageVectors`, `QueryContext`
* `config.yaml` — Non-secret configuration (models, weights, dimensions)
* `.env.example` — Credential templates

### References

* `.copilot-tracking/research/2026-02-26/model-strategy-index-config-research.md` — Multimodal model research: Florence (Q2), Cohere v4 (Q3), architecture options (Q4)
* `.copilot-tracking/research/2026-02-26/ai-search-pipeline-research.md` — Original pipeline research
* `.copilot-tracking/plans/logs/2026-02-26/ai-search-pipeline-log.md` — Original planning log with deviations and follow-on work

### Service Tier

Target: **S2 Standard** — supports ~1 TB vector storage, sufficient for current `image_vector` addition (~37 GB at 10M docs). S2 tier is a provisioning decision, not a code change. Document in README for operational guidance.

### Standards References

* `requirements.md` Section 12 — Mandatory rules (no hardcoded secrets, Azure Foundry only, config externalized)
* **Foundry-only waiver**: Section 12 states "Azure Foundry only for models." Florence is a separate Azure Cognitive Services resource, not a Foundry catalog model. The user explicitly selected Florence as the primary image embedding model and confirmed they will provision the required Azure services. This deviation is documented as DD-01 in the Planning Log.

## Implementation Checklist

### [x] Implementation Phase 1: Configuration & Secrets

<!-- parallelizable: false -->

* [x] Step 1.1: Add Florence secrets model and config extensions to `config.py`
  * Details: .copilot-tracking/details/2026-02-26/multimodal-embeddings-details.md (Lines 16-105)
* [x] Step 1.2: Update `config.yaml` with image embedding settings and scoring profile config
  * Details: .copilot-tracking/details/2026-02-26/multimodal-embeddings-details.md (Lines 107-148)
* [x] Step 1.3: Update `.env.example` with Florence credentials
  * Details: .copilot-tracking/details/2026-02-26/multimodal-embeddings-details.md (Lines 150-167)

### [x] Implementation Phase 2: Client & Embedding Layer

<!-- parallelizable: true -->

* [x] Step 2.1: Add Florence client factory to `clients.py`
  * Details: .copilot-tracking/details/2026-02-26/multimodal-embeddings-details.md (Lines 173-211)
* [x] Step 2.2: Create `embeddings/image.py` — image embedding module
  * Details: .copilot-tracking/details/2026-02-26/multimodal-embeddings-details.md (Lines 213-327)
* [x] Step 2.3: Update `embeddings/pipeline.py` to include image embedding
  * Details: .copilot-tracking/details/2026-02-26/multimodal-embeddings-details.md (Lines 329-388)

### [x] Implementation Phase 3: Index Schema & Models

<!-- parallelizable: true -->

* [x] Step 3.1: Add `image_vector` to `SearchDocument` and `ImageVectors` models
  * Details: .copilot-tracking/details/2026-02-26/multimodal-embeddings-details.md (Lines 394-433)
* [x] Step 3.2: Add `image_vector` field and scoring profile to `indexing/schema.py`
  * Details: .copilot-tracking/details/2026-02-26/multimodal-embeddings-details.md (Lines 435-504)

### [x] Implementation Phase 4: Retrieval Layer

<!-- parallelizable: true -->

* [x] Step 4.1: Update `retrieval/query.py` to support image query vectors
  * Details: .copilot-tracking/details/2026-02-26/multimodal-embeddings-details.md (Lines 510-574)
* [x] Step 4.2: Update `retrieval/search.py` to include `image_vector` in hybrid search
  * Details: .copilot-tracking/details/2026-02-26/multimodal-embeddings-details.md (Lines 576-628)

### [x] Implementation Phase 5: Dependencies & Documentation

<!-- parallelizable: true -->

* [x] Step 5.1: Verify `httpx` async dependency in `pyproject.toml`
  * Details: .copilot-tracking/details/2026-02-26/multimodal-embeddings-details.md (Lines 634-646)
* [x] Step 5.2: Update `README.md` with image embedding setup and S2 tier guidance
  * Details: .copilot-tracking/details/2026-02-26/multimodal-embeddings-details.md (Lines 648-664)

### [x] Implementation Phase 6: Tests

<!-- parallelizable: true -->

* [x] Step 6.1: Add unit tests for `embeddings/image.py`
  * Details: .copilot-tracking/details/2026-02-26/multimodal-embeddings-details.md (Lines 670-709)
* [x] Step 6.2: Update existing embedding pipeline test for image vector
  * Details: .copilot-tracking/details/2026-02-26/multimodal-embeddings-details.md (Lines 711-723)
* [x] Step 6.3: Add unit tests for updated search and query modules
  * Details: .copilot-tracking/details/2026-02-26/multimodal-embeddings-details.md (Lines 725-745)

### [x] Implementation Phase 7: Validation

<!-- parallelizable: false -->

* [x] Step 7.1: Run full project validation
  * Execute `uv run ruff check src/ tests/`
  * Execute `uv run mypy src/`
  * Execute `uv run pytest tests/ -m "not integration"`
* [x] Step 7.2: Fix minor validation issues
  * Iterate on lint errors and build warnings
  * Apply fixes directly when corrections are straightforward
* [x] Step 7.3: Report blocking issues
  * Document issues requiring additional research
  * Provide user with next steps and recommended planning

## Planning Log

See [multimodal-embeddings-log.md](.copilot-tracking/plans/logs/2026-02-26/multimodal-embeddings-log.md) for discrepancy tracking, implementation paths considered, and suggested follow-on work.

## Dependencies

* Python >= 3.11
* UV (package manager)
* `httpx>=0.27` — async HTTP client for Florence REST API (already in deps)
* `openai>=1.58.0` — existing Azure OpenAI client
* `azure-search-documents>=11.6.0` — existing Azure AI Search SDK
* Azure Computer Vision 4.0 resource — provisioned in same region as Foundry

## Success Criteria

* `embed_image()` returns 1024-dim vector from Florence `vectorizeImage` endpoint — Traces to: user requirement for image embeddings
* `embed_text_for_image_search()` returns 1024-dim vector from Florence `vectorizeText` endpoint — Traces to: cross-modal search capability
* Azure AI Search index includes `image_vector` field with HNSW cosine at 1024 dims — Traces to: index enhancement
* Hybrid search includes `image_vector` with configurable `image_weight` — Traces to: weighted retrieval
* Scoring profile boosts `generation_prompt` and `tags` in BM25 scoring — Traces to: user-selected index improvement
* `uv run ruff check src/ tests/` passes with zero errors — Traces to: code quality
* `uv run mypy src/` passes with zero errors — Traces to: type safety
* `uv run pytest tests/ -m "not integration"` passes all unit tests — Traces to: functional correctness
* Config loads Florence credentials from `.env` and image dimensions from `config.yaml` — Traces to: requirements.md Section 12
