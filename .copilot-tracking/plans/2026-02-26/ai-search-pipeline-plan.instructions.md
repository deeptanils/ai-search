---
applyTo: '.copilot-tracking/changes/2026-02-26/ai-search-pipeline-changes.md'
---
<!-- markdownlint-disable-file -->
# Implementation Plan: Candidate Generation & AI Search Pipeline

## Overview

Build a production-grade Python pipeline that accepts images with generation prompts, extracts structured metadata and multi-dimensional descriptions via GPT-4o, encodes multi-vector embeddings via text-embedding-3-large, indexes into Azure AI Search, and supports hybrid multi-vector retrieval with application-level re-ranking and diversity.

## Objectives

### User Requirements

* Accept image + generation prompt as input and generate synthetic metadata via LLM — Source: requirements.md Section 5
* Extract structured narrative, emotional, structural, and character information from images — Source: requirements.md Sections 4, 6
* Generate multi-vector embeddings (semantic, structural, style, character sub-vectors) — Source: requirements.md Section 7
* Index into Azure AI Search with HNSW + cosine similarity and hybrid search — Source: requirements.md Sections 8-9
* Support hybrid retrieval with configurable weighted scoring and candidate re-ranking — Source: requirements.md Sections 7, 9
* Python project managed with UV — Source: requirements.md Section 10
* All models served through Azure AI Foundry only — Source: requirements.md Section 2.1
* No hardcoded secrets; externalize config via `.env` + `config.yaml` — Source: requirements.md Sections 3, 12

### Derived Objectives

* Use `openai` SDK with `AzureOpenAI` client for all Azure AI Foundry calls — Derived from: Azure AI Foundry constraint + SDK maturity research
* Use single structured GPT-4o vision call per image for all extraction — Derived from: Cost/latency optimization; single call minimizes token charges and latency
* Use Matryoshka dimension reduction (text-embedding-3-large) for all vector types — Derived from: Azure AI Foundry-only constraint eliminates DINOv2/Style Encoder
* Flatten character sub-vectors to top-level index fields (3 character slots) — Derived from: Azure AI Search does not support vector search inside `Collection(Edm.ComplexType)`
* Map config weights × 10 for vector query `weight` parameters to preserve ratio with BM25's implicit 1.0 — Derived from: No direct BM25 weight parameter in RRF
* Implement three-stage retrieval: Azure AI Search (RRF) → rule-based re-rank → MMR diversity — Derived from: P95 < 300ms latency constraint rules out LLM re-ranking in real-time path
* Use hatchling build backend with src-layout (`packages = ["src/ai_search"]`) — Derived from: UV default, modern Python standard

## Context Summary

### Project Files

* requirements.md - Comprehensive requirements document (v1 + v2) defining pipeline stages, Azure constraints, index schema, retrieval strategy, and project structure

### References

* .copilot-tracking/research/2026-02-26/ai-search-pipeline-research.md - Consolidated research with 9 key discoveries and selected approach
* .copilot-tracking/research/subagents/2026-02-26/azure-ai-foundry-sdk-research.md - OpenAI SDK patterns (Lines 38-180), vision API (Lines 300-430), batch processing (Lines 430-580)
* .copilot-tracking/research/subagents/2026-02-26/azure-ai-search-sdk-research.md - Index schema (Lines 65-200), vector config (Lines 210-290), hybrid search (Lines 310-440), batch indexing (Lines 490-620)
* .copilot-tracking/research/subagents/2026-02-26/uv-python-project-research.md - pyproject.toml (Lines 70-160), project structure (Lines 170-280), config management (Lines 290-400), bootstrap sequence (Lines 540-580)
* .copilot-tracking/research/subagents/2026-02-26/multi-vector-encoding-research.md - Dimension strategy (Lines 80-140), character vectors (Lines 300-420), unified pipeline architecture (Lines 440-550)
* .copilot-tracking/research/subagents/2026-02-26/hybrid-retrieval-research.md - RRF weight mapping (Lines 70-170), three-stage retrieval (Lines 330-500), MMR (Lines 600-720)

### Standards References

* requirements.md Section 12 - Mandatory rules (no hardcoded secrets, Azure Foundry only, config externalized, index schema versioned)

## Implementation Checklist

### [x] Implementation Phase 1: Project Scaffolding

<!-- parallelizable: false -->

* [x] Step 1.1: Initialize UV project and create directory structure
  * Details: .copilot-tracking/details/2026-02-26/ai-search-pipeline-details.md (Lines 12-43)
* [x] Step 1.2: Create pyproject.toml with all dependencies and tooling config
  * Details: .copilot-tracking/details/2026-02-26/ai-search-pipeline-details.md (Lines 45-134)
* [x] Step 1.3: Create config.yaml, .env.example, .gitignore, and README.md
  * Details: .copilot-tracking/details/2026-02-26/ai-search-pipeline-details.md (Lines 136-253)
* [x] Step 1.4: Create all `__init__.py` files for package structure
  * Details: .copilot-tracking/details/2026-02-26/ai-search-pipeline-details.md (Lines 255-287)
* [x] Step 1.5: Run `uv sync` and verify import works
  * Details: .copilot-tracking/details/2026-02-26/ai-search-pipeline-details.md (Lines 289-306)

### [x] Implementation Phase 2: Configuration & Shared Models

<!-- parallelizable: false -->

* [x] Step 2.1: Create config.py with pydantic-settings + YAML loader
  * Details: .copilot-tracking/details/2026-02-26/ai-search-pipeline-details.md (Lines 312-504)
* [x] Step 2.2: Create models.py with shared Pydantic data models
  * Details: .copilot-tracking/details/2026-02-26/ai-search-pipeline-details.md (Lines 506-680)
* [x] Step 2.3: Create Azure AI Foundry client factory (sync + async)
  * Details: .copilot-tracking/details/2026-02-26/ai-search-pipeline-details.md (Lines 682-767)
* [x] Step 2.4: Create Azure AI Search client factory
  * Details: .copilot-tracking/details/2026-02-26/ai-search-pipeline-details.md (Lines 769-778)

### [x] Implementation Phase 3: Ingestion & Extraction

<!-- parallelizable: true -->

* [x] Step 3.1: Create ingestion/loader.py — image loading (URL + binary)
  * Details: .copilot-tracking/details/2026-02-26/ai-search-pipeline-details.md (Lines 784-841)
* [x] Step 3.2: Create ingestion/metadata.py — LLM synthetic metadata generation
  * Details: .copilot-tracking/details/2026-02-26/ai-search-pipeline-details.md (Lines 843-906)
* [x] Step 3.3: Create extraction module — unified GPT-4o vision extraction
  * Details: .copilot-tracking/details/2026-02-26/ai-search-pipeline-details.md (Lines 908-992)
* [x] Step 3.4: Create extraction sub-modules (narrative, emotion, objects, low_light)
  * Details: .copilot-tracking/details/2026-02-26/ai-search-pipeline-details.md (Lines 994-1029)

### [x] Implementation Phase 4: Embedding Generation

<!-- parallelizable: true -->

* [x] Step 4.1: Create embeddings/encoder.py — base embedding service with Matryoshka dimensions
  * Details: .copilot-tracking/details/2026-02-26/ai-search-pipeline-details.md (Lines 1035-1107)
* [x] Step 4.2: Create embeddings/semantic.py, structural.py, style.py — typed wrappers
  * Details: .copilot-tracking/details/2026-02-26/ai-search-pipeline-details.md (Lines 1109-1146)
* [x] Step 4.3: Create embeddings/character.py — per-character sub-vector generation
  * Details: .copilot-tracking/details/2026-02-26/ai-search-pipeline-details.md (Lines 1148-1209)
* [x] Step 4.4: Create embeddings/pipeline.py — orchestrator that groups by dimension and parallelizes
  * Details: .copilot-tracking/details/2026-02-26/ai-search-pipeline-details.md (Lines 1211-1261)

### [x] Implementation Phase 5: Azure AI Search Indexing

<!-- parallelizable: true -->

* [x] Step 5.1: Create indexing/schema.py — full index definition with all fields + HNSW config
  * Details: .copilot-tracking/details/2026-02-26/ai-search-pipeline-details.md (Lines 1267-1293)
* [x] Step 5.2: Create indexing/indexer.py — batch document upload with retry logic
  * Details: .copilot-tracking/details/2026-02-26/ai-search-pipeline-details.md (Lines 1295-1321)
* [x] Step 5.3: Create indexing/cli.py — CLI entry point for index creation and document upload
  * Details: .copilot-tracking/details/2026-02-26/ai-search-pipeline-details.md (Lines 1323-1368)

### [x] Implementation Phase 6: Retrieval Service

<!-- parallelizable: true -->

* [x] Step 6.1: Create retrieval/query.py — query embedding generation (text + image queries)
  * Details: .copilot-tracking/details/2026-02-26/ai-search-pipeline-details.md (Lines 1374-1395)
* [x] Step 6.2: Create retrieval/search.py — hybrid search with configurable weights
  * Details: .copilot-tracking/details/2026-02-26/ai-search-pipeline-details.md (Lines 1397-1421)
* [x] Step 6.3: Create retrieval/reranker.py — Stage 2 rule-based re-ranking
  * Details: .copilot-tracking/details/2026-02-26/ai-search-pipeline-details.md (Lines 1423-1447)
* [x] Step 6.4: Create retrieval/diversity.py — Stage 3 MMR diversity
  * Details: .copilot-tracking/details/2026-02-26/ai-search-pipeline-details.md (Lines 1449-1471)
* [x] Step 6.5: Create retrieval/pipeline.py — three-stage orchestrator
  * Details: .copilot-tracking/details/2026-02-26/ai-search-pipeline-details.md (Lines 1473-1496)

### [x] Implementation Phase 7: CLI Entry Points

<!-- parallelizable: true -->

* [x] Step 7.1: Create ingestion/cli.py — ingest command
  * Details: .copilot-tracking/details/2026-02-26/ai-search-pipeline-details.md (Lines 1502-1518)
* [x] Step 7.2: Create retrieval/cli.py — query command
  * Details: .copilot-tracking/details/2026-02-26/ai-search-pipeline-details.md (Lines 1520-1537)

### [x] Implementation Phase 8: Tests

<!-- parallelizable: true -->

* [x] Step 8.1: Create tests/conftest.py with shared fixtures and mocks
  * Details: .copilot-tracking/details/2026-02-26/ai-search-pipeline-details.md (Lines 1543-1565)
* [x] Step 8.2: Create unit tests for config, models, and client factories
  * Details: .copilot-tracking/details/2026-02-26/ai-search-pipeline-details.md (Lines 1567-1584)
* [x] Step 8.3: Create unit tests for extraction and embedding modules
  * Details: .copilot-tracking/details/2026-02-26/ai-search-pipeline-details.md (Lines 1586-1605)
* [x] Step 8.4: Create unit tests for indexing schema and retrieval pipeline
  * Details: .copilot-tracking/details/2026-02-26/ai-search-pipeline-details.md (Lines 1607-1629)
* [x] Step 8.5: Create integration test markers and fixtures
  * Details: .copilot-tracking/details/2026-02-26/ai-search-pipeline-details.md (Lines 1631-1645)

### [x] Implementation Phase 9: Validation

<!-- parallelizable: false -->

* [x] Step 9.1: Run full project validation
  * Execute `uv run ruff check src/ tests/`
  * Execute `uv run mypy src/`
  * Execute `uv run pytest tests/ -m "not integration"`
* [x] Step 9.2: Fix minor validation issues
  * Iterate on lint errors and build warnings
  * Apply fixes directly when corrections are straightforward
* [x] Step 9.3: Report blocking issues
  * Document issues requiring additional research
  * Provide user with next steps and recommended planning
  * Avoid large-scale fixes within this phase

## Planning Log

See [ai-search-pipeline-log.md](.copilot-tracking/plans/logs/2026-02-26/ai-search-pipeline-log.md) for discrepancy tracking, implementation paths considered, and suggested follow-on work.

## Dependencies

* Python >= 3.11
* UV (package manager)
* `openai>=1.58.0` — Azure AI Foundry LLM + embeddings
* `azure-search-documents>=11.6.0` — Azure AI Search index + hybrid search
* `azure-identity>=1.17.0` — Azure Entra ID auth (production)
* `pydantic>=2.0` — Data models + structured output schemas
* `pydantic-settings>=2.0` — Settings from .env / env vars
* `pyyaml>=6.0` — YAML config parsing
* `python-dotenv>=1.0` — .env file loading
* `pillow>=10.0` — Image processing
* `httpx>=0.27` — Async HTTP client
* `structlog>=24.0` — Structured logging
* `numpy>=1.26` — MMR diversity computation
* Dev: `pytest>=8.0`, `pytest-asyncio>=0.23`, `pytest-cov>=5.0`, `ruff>=0.5`, `mypy>=1.10`

## Success Criteria

* `uv run python -c "import ai_search"` succeeds — Traces to: project scaffolding
* `uv run ruff check src/ tests/` passes with zero errors — Traces to: code quality
* `uv run mypy src/` passes with zero errors — Traces to: type safety
* `uv run pytest tests/ -m "not integration"` passes all unit tests — Traces to: functional correctness
* Config loads from `.env` + `config.yaml` with pydantic validation — Traces to: requirements.md Section 3
* Azure AI Search index schema creates successfully with all vector fields + HNSW — Traces to: requirements.md Section 8
* Single GPT-4o vision call returns structured extraction matching Pydantic models — Traces to: extraction architecture
* Embedding pipeline generates vectors at correct dimensions (3072/1024/512/256) — Traces to: requirements.md Section 7
* Hybrid search returns results using BM25 + 3 vector queries with configurable weights — Traces to: requirements.md Section 9
* Re-ranking pipeline applies emotional/narrative/object/low-light scoring — Traces to: requirements.md Section 7.2
* MMR diversity produces non-redundant top-N results — Traces to: retrieval quality
