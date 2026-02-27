<!-- markdownlint-disable-file -->
# Release Changes: Review Rework — Major Findings + Observability

**Related Plan**: review-rework-plan.instructions.md
**Implementation Date**: 2026-02-26

## Summary

Address 3 major review findings (RPI-004, IV-011, IV-014) and add structured logging observability to Florence validation paths and secrets guard.

## Changes

### Added

* `tests/test_retrieval/test_query.py` — 3 async tests for `generate_query_vectors()`: return keys, image URL branch, text-only branch (RPI-004)
* `tests/test_embeddings/test_image.py` — 6 new validation tests: `TestEmbedImageValidation` (3 tests), `TestEmbedTextForImageSearchValidation` (2 tests), `TestSecretsValidation` (1 test)

### Modified

* `src/ai_search/config.py` — Changed `AzureComputerVisionSecrets` defaults from `str = ""` to `str | None = None` for `endpoint` and `api_key` (IV-014)
* `src/ai_search/clients.py` — Added `structlog` import, `logger` instance, and validation guard in `get_cv_client()` that raises `ValueError` with actionable error message when secrets are `None`; logs `error` via structlog before raising (IV-014 + observability)
* `src/ai_search/embeddings/image.py` — Added response payload validation to `embed_image()` and `embed_text_for_image_search()`: `data.get("vector")` + 1024-dim check + `ValueError` with diagnostic info; logs `warning` via structlog before raising (IV-011 + observability). Added `result: list[float]` type annotation to satisfy mypy `no-any-return` rule.

### Removed

## Additional or Deviating Changes

* mypy `no-any-return` fix: Plan specified `return vector` after validation but `data.get()` returns `Any`. Added `result: list[float] = vector` assignment after the guard to satisfy mypy without casting.
  * Reason: Plan's code snippet did not account for mypy strict mode behavior on `.get()` return type.
* Test count exceeded plan estimate: Plan estimated 63+ tests (57 + 6). Actual result is 66 passed (57 existing + 9 new: 6 validation + 3 query tests).
  * Reason: Plan counted 6 new tests for Phases 2-3 combined but the 3 query tests were separately tallied; the success criteria total was underspecified.

## Release Summary

Total files affected: 5 (3 modified, 1 created, 1 test file extended)

**Files created:**
* `tests/test_retrieval/test_query.py` — 3 tests for `generate_query_vectors()` image embedding branches

**Files modified:**
* `src/ai_search/config.py` — `AzureComputerVisionSecrets` secrets defaults changed to `None`
* `src/ai_search/clients.py` — Validation guard with structured error logging in `get_cv_client()`
* `src/ai_search/embeddings/image.py` — Florence response validation with structured warning logging
* `tests/test_embeddings/test_image.py` — 6 validation tests added (3 embed_image, 2 embed_text, 1 secrets)

**Validation results:**
* `uv run ruff check .` — All checks passed
* `uv run mypy src/` — Success: no issues found in 33 source files
* `uv run pytest -v` — 66 passed, 3 skipped in 2.53s
