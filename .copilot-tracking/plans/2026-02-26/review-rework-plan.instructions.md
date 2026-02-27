---
applyTo: '.copilot-tracking/changes/2026-02-26/review-rework-changes.md'
---
<!-- markdownlint-disable-file -->
# Implementation Plan: Review Rework — Major Findings + Observability

## Overview

Address three major review findings (RPI-004, IV-011, IV-014) with structured logging observability for Florence validation paths and secrets guard failures.

## Objectives

### User Requirements

* RPI-004: Create missing `tests/test_retrieval/test_query.py` with tests for `generate_query_vectors()` — Source: Review log RPI-004
* IV-011: Add Florence API response payload validation to prevent `KeyError` and silent data corruption — Source: Review log IV-011
* IV-014: Fix `AzureComputerVisionSecrets` empty-string defaults to fail fast on missing credentials — Source: Review log IV-014
* Add observability and tracking to changed code paths — Source: User request ("also consider observability and tracking")

### Derived Objectives

* Add structured logging (`structlog`) to validation branches in `embed_image()` and `embed_text_for_image_search()` for production diagnostics — Derived from: Observability request + IV-011 validation paths creating new error branches that should be observable
* Log secrets-guard failure in `get_cv_client()` before raising — Derived from: Observability request + IV-014 guard creating a new error path that should be observable
* Verify zero regressions in existing 57 tests — Derived from: Rework scope must not break existing functionality

## Context Summary

### Project Files

* `src/ai_search/embeddings/image.py` (93 lines) — Florence embedding functions; unvalidated `response.json()["vector"]` on lines 62 and 92
* `src/ai_search/config.py` (196 lines) — `AzureComputerVisionSecrets` with `str = ""` defaults on lines 70-71
* `src/ai_search/clients.py` (79 lines) — `get_cv_client()` on lines 73-79; no validation guard before client construction
* `src/ai_search/retrieval/query.py` (93 lines) — `generate_query_vectors()` lines 31-84; branches on line 78 for image URL vs text
* `tests/test_embeddings/test_image.py` (99 lines) — 5 existing tests with `mock_cv_client` fixture pattern
* `tests/test_retrieval/test_query.py` — MISSING; needs creation

### References

* `.copilot-tracking/research/2026-02-26/review-rework-major-findings-research.md` — Consolidated research with selected approaches
* `.copilot-tracking/reviews/2026-02-26/multimodal-embeddings-plan-review.md` — Review log with 3 major findings

### Standards References

* `requirements.md` Section 12 — Error handling and logging conventions
* Commit message instructions — Conventional Commits with emoji footer

## Implementation Checklist

### [x] Implementation Phase 1: Secrets Hardening (IV-014)

<!-- parallelizable: false -->

* [x] Step 1.1: Update `AzureComputerVisionSecrets` defaults to `str | None = None`
  * Details: .copilot-tracking/details/2026-02-26/review-rework-details.md (Lines 15-36)
* [x] Step 1.2: Add validation guard with structured logging in `get_cv_client()`
  * Details: .copilot-tracking/details/2026-02-26/review-rework-details.md (Lines 38-73)
* [x] Step 1.3: Validate phase changes
  * Run `uv run ruff check src/ai_search/config.py src/ai_search/clients.py`
  * Run `uv run mypy src/ai_search/config.py src/ai_search/clients.py`

### [x] Implementation Phase 2: Response Validation + Observability (IV-011)

<!-- parallelizable: false -->

Depends on Phase 1 (secrets type changes affect `get_cv_client()` callers).

* [x] Step 2.1: Add response validation with structured logging to `embed_image()`
  * Details: .copilot-tracking/details/2026-02-26/review-rework-details.md (Lines 78-120)
* [x] Step 2.2: Add response validation with structured logging to `embed_text_for_image_search()`
  * Details: .copilot-tracking/details/2026-02-26/review-rework-details.md (Lines 122-157)
* [x] Step 2.3: Add 6 validation tests to `test_image.py`
  * Details: .copilot-tracking/details/2026-02-26/review-rework-details.md (Lines 159-220)
* [x] Step 2.4: Validate phase changes
  * Run `uv run ruff check src/ai_search/embeddings/image.py tests/test_embeddings/test_image.py`
  * Run `uv run mypy src/ai_search/embeddings/image.py`
  * Run `uv run pytest tests/test_embeddings/test_image.py -v`

### [x] Implementation Phase 3: Missing Query Tests (RPI-004)

<!-- parallelizable: true -->

Independent of Phases 1-2 (new file, no shared test state).

* [x] Step 3.1: Create `tests/test_retrieval/test_query.py` with 3 tests
  * Details: .copilot-tracking/details/2026-02-26/review-rework-details.md (Lines 225-310)
* [x] Step 3.2: Validate phase changes
  * Run `uv run ruff check tests/test_retrieval/test_query.py`
  * Run `uv run pytest tests/test_retrieval/test_query.py -v`

### [x] Implementation Phase 4: Validation

<!-- parallelizable: false -->

* [x] Step 4.1: Run full project validation
  * Execute `uv run ruff check .`
  * Execute `uv run mypy src/`
  * Execute `uv run pytest -v` — expect 63+ tests (57 existing + 6 new)
* [x] Step 4.2: Fix minor validation issues
  * Iterate on lint errors and build warnings
  * Apply fixes directly when corrections are straightforward
* [ ] Step 4.3: Report blocking issues
  * Document issues requiring additional research
  * Provide user with next steps and recommended planning
  * Avoid large-scale fixes within this phase

## Planning Log

See [review-rework-log.md](.copilot-tracking/plans/logs/2026-02-26/review-rework-log.md) for discrepancy tracking, implementation paths considered, and suggested follow-on work.

## Dependencies

* Python 3.12 with UV package manager
* structlog (already installed and imported in `image.py`)
* httpx (already installed)
* pytest + pytest-asyncio (already installed)
* ruff, mypy (already installed)

## Success Criteria

* `tests/test_retrieval/test_query.py` exists with 3 passing tests covering image_vector key, image URL branch, and text-only branch — Traces to: RPI-004
* `embed_image()` and `embed_text_for_image_search()` raise `ValueError` on missing key, empty vector, or wrong dimensions — Traces to: IV-011
* `AzureComputerVisionSecrets` uses `str | None = None` defaults; `get_cv_client()` raises `ValueError` when endpoint/api_key is `None` — Traces to: IV-014
* All validation failures logged via `structlog` before raising exceptions — Traces to: Observability requirement
* `uv run ruff check .` passes clean — Traces to: Project standards
* `uv run mypy src/` passes clean — Traces to: Project standards
* `uv run pytest -v` passes with 63+ tests (57 existing + 6+ new) and zero regressions — Traces to: No-regression requirement
