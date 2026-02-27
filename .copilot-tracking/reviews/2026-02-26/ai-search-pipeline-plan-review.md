<!-- markdownlint-disable-file -->
# Review Log: Candidate Generation & AI Search Pipeline

## Metadata

| Field | Value |
|-------|-------|
| **Review Date** | 2026-02-26 |
| **Plan** | `.copilot-tracking/plans/2026-02-26/ai-search-pipeline-plan.instructions.md` |
| **Changes Log** | `.copilot-tracking/changes/2026-02-26/ai-search-pipeline-changes.md` |
| **Research** | `.copilot-tracking/research/2026-02-26/ai-search-pipeline-research.md` |
| **Planning Log** | `.copilot-tracking/plans/logs/2026-02-26/ai-search-pipeline-log.md` |
| **Details** | `.copilot-tracking/details/2026-02-26/ai-search-pipeline-details.md` |
| **RPI Validations** | `.copilot-tracking/reviews/rpi/2026-02-26/ai-search-pipeline-plan-{001..009}-validation.md` |
| **Impl Validation** | `.copilot-tracking/reviews/logs/2026-02-26/ai-search-pipeline-impl-validation.md` |
| **Status** | ⚠️ Needs Rework |

## Severity Counts

| Severity | Count |
|----------|-------|
| Critical | 1 |
| Major | 7 |
| Minor | 11 |
| **Total** | **19** |

## Phase Validation Summary (RPI)

All 9 phases passed RPI validation — every plan checklist item has a corresponding implemented file with matching content.

| Phase | Description | Status | Findings |
|-------|-------------|--------|----------|
| 1 | Project Scaffolding | ✅ Pass | 1 minor (additive `.gitignore` entry) |
| 2 | Configuration & Shared Models | ✅ Pass | 0 (2 informational) |
| 3 | Ingestion & Extraction | ✅ Pass | 2 minor (no tests for accessor sub-modules and `metadata.py`) |
| 4 | Embedding Generation | ✅ Pass | 0 (3 informational) |
| 5 | Azure AI Search Indexing | ✅ Pass | 0 (3 informational) |
| 6 | Retrieval Service | ✅ Pass | 0 (4 low/informational) |
| 7 | CLI Entry Points | ✅ Pass | 0 (4 low/informational) |
| 8 | Tests | ✅ Pass | 2 minor (test count discrepancies in changes log) |
| 9 | Validation | ✅ Pass | 0 (3 informational) |

## RPI Validation Findings

### Phase 1-5: Fully Verified

All scaffolding, configuration, ingestion/extraction, embedding, and indexing files match plan specifications. Deviations DD-02 (unified extraction) and DD-03 (embedding grouping) are correctly implemented and documented in the planning log.

### Phase 6-7: Fully Verified

Retrieval pipeline (5 modules) and CLI entry points (2 modules) match plan specifications. Deviation DD-04 (rule-based re-ranking only) correctly applied. Low-priority observations: MMR precomputes full similarity matrix (memory concern at scale), query vectors use sync LLM calls inside async function, CLI modules lack Azure exception handling.

### Phase 8-9: Fully Verified

Test suite (50 passed, 3 deselected) matches plan requirements. Minor discrepancies in changes log test counts vs actual (changes log says 10 in `test_models.py`, actual is 11; says 12 in `test_reranker.py`, actual is 14). Validation tools (ruff, mypy, pytest) all pass clean.

## Implementation Quality Findings

Full-quality implementation validation identified 17 findings across 37 source and 16 test files. See `.copilot-tracking/reviews/logs/2026-02-26/ai-search-pipeline-impl-validation.md` for complete details.

### Critical (1)

* **IV-009** (Architecture): `generate_query_vectors()` in `retrieval/query.py` is `async def` but calls `client.chat.completions.create()` synchronously for structural/style LLM descriptions, blocking the event loop for 2-10s per query. Only the embedding calls use `await asyncio.gather()`. Fix: use `await client.chat.completions.create()` with `AsyncAzureOpenAI`, or move LLM calls to a thread executor.

### Major (7)

* **IV-004** (Design): `lru_cache` on client factory functions in `clients.py` caches SDK clients indefinitely with no `close()` or cache invalidation mechanism. Risk: resource leaks in long-running processes.
* **IV-005** (Design): `rerank_candidates()` in `reranker.py` mutates the input list in-place via `list.sort()`. Callers may not expect side effects. Fix: use `sorted()` to return a new list.
* **IV-006** (Architecture): `asyncio.run()` wrappers (`generate_query_vectors_sync`, `retrieve_sync`) fail with `RuntimeError` if called inside an existing event loop (e.g., Jupyter, FastAPI). Fix: use `asyncio.get_event_loop().run_until_complete()` with loop existence check, or document the constraint.
* **IV-012** (Error Handling): No retry logic for OpenAI API calls in `extraction/extractor.py`. Single transient failure aborts the entire extraction.
* **IV-013** (Error Handling): No retry logic for OpenAI embedding API calls in `embeddings/encoder.py`. Batch processing of many images amplifies failure probability.
* **IV-014** (Architecture): **MMR diversity stage is functionally inert.** `SELECT_FIELDS` in `search.py` does not include `semantic_vector`, so `_extract_embeddings()` in `pipeline.py` always falls back to zero vectors. All documents appear identical to the MMR algorithm, making Stage 3 a no-op.
* **IV-015** (Test Coverage): No unit tests for `retrieve()` orchestrator or `generate_query_vectors()` function — the two most critical runtime paths.

### Minor (9)

* **IV-001** (DRY): Extraction sub-modules (`narrative.py`, `emotion.py`, `objects.py`, `low_light.py`) import `ImageExtraction` but only re-export accessor functions. Dead import in each module.
* **IV-002** (DRY): `metadata.py` imports `ImageInput` but only uses `AzureOpenAI` and `load_config`.
* **IV-003** (Architecture): `ingestion/cli.py` imports from `extraction` and `embeddings` packages, crossing module boundaries (ingestion → extraction → embeddings).
* **IV-007** (DRY): `semantic.py`, `structural.py`, `style.py` are nearly identical thin wrappers with only dimension values differing.
* **IV-008** (Design): Hardcoded 9 character vector field names in `schema.py` and `indexer.py` coupled to `MAX_CHARACTERS = 3` config.
* **IV-010** (Design): Hardcoded `detail: "high"` in `ImageInput.to_openai_image_content()` — not configurable.
* **IV-011** (Design): Hardcoded JPEG MIME type assumption in `ImageInput.from_file()`.
* **IV-016** (Security): OData filter string passed unsanitized to Azure Search `filter` parameter.
* **IV-017** (General): `structlog` is a dependency but `structlog.configure()` is never called — logs use default configuration.

## Validation Command Results

| Command | Result | Details |
|---------|--------|---------|
| `uv run ruff check src/ tests/` | ✅ Pass | All checks passed |
| `uv run mypy src/` | ✅ Pass | No issues in 32 source files |
| `uv run pytest tests/ -m "not integration"` | ✅ Pass | 50 passed, 3 deselected, 0.54s |
| VS Code Pylance diagnostics | ⚠️ Info | 4 unresolved imports (azure.*, openai) — venv not selected in IDE, not a real issue |

## Missing Work and Deviations

### From Plan (VF Findings)

* **VF-01** (Major, resolved): Plan Phase 8 line references to details file were out of range — implementation proceeded correctly regardless.
* **VF-04** (Minor): No explicit success criterion for "Azure AI Foundry only" or "no hardcoded secrets" enforcement.
* **VF-05** (Minor): No schema versioning mechanism per requirements.md Section 12.
* **VF-06** (Minor): Incremental re-indexing (req v1 Section 8) and observability (req v1 Section 9) not addressed or explicitly deferred.

### From Implementation (ID Findings)

* **ID-01**: `type: ignore[misc, list-item]` on OpenAI multimodal message dicts — known SDK limitation, no runtime impact.
* **ID-02**: `type: ignore[arg-type]` on Azure Search `VectorizedQuery` — known SDK typing inconsistency, no runtime impact.
* **ID-03**: `SearchableField`/`SimpleField` factory functions typed as `list[SearchField]` — correct resolution.

## Follow-Up Work

### Deferred from Scope (Planning Log)

15 items tracked in planning log (WI-01 through WI-15). Key items:

| ID | Priority | Description |
|----|----------|-------------|
| WI-01 | High | Azure OpenAI Batch API for 10M+ scale processing |
| WI-02 | High | Evaluation framework (NDCG@K, MRR, precision@K) |
| WI-03 | Medium | LLM re-ranking for premium/offline tier |
| WI-05 | Medium | Azure Entra ID authentication |
| WI-09 | Medium | CI/CD pipeline with UV |
| WI-11 | Medium | Structured logging configuration |
| WI-12 | Medium | Image query support in retrieval CLI |
| WI-14 | Low | Index schema versioning |
| WI-15 | Medium | End-to-end integration tests |

### Discovered During Review

| ID | Priority | Description | Source |
|----|----------|-------------|--------|
| RW-01 | **Critical** | Fix sync-in-async blocking in `generate_query_vectors()` — use `AsyncAzureOpenAI` for LLM calls | IV-009 |
| RW-02 | **Critical** | Add `semantic_vector` to `SELECT_FIELDS` in `search.py` to enable MMR diversity | IV-014 |
| RW-03 | Major | Add retry logic for OpenAI API calls in `extractor.py` and `encoder.py` | IV-012, IV-013 |
| RW-04 | Major | Add unit tests for `retrieve()` and `generate_query_vectors()` | IV-015 |
| RW-05 | Major | Fix `rerank_candidates()` to not mutate input list | IV-005 |
| RW-06 | Major | Add client cleanup mechanism for `lru_cache` factories | IV-004 |
| RW-07 | Major | Handle `asyncio.run()` inside existing event loops | IV-006 |
| RW-08 | Minor | Configure `structlog` formatters and processors | IV-017 |
| RW-09 | Minor | Correct test count discrepancies in changes log | VF-P8-02, VF-P8-03 |

## Overall Status

**Status**: ⚠️ Needs Rework

**Reviewer Notes**:

The implementation is **architecturally sound and functionally complete** against all 9 plan phases (38/38 checklist items verified). Code quality is high: clean module boundaries, strong Pydantic contracts, proper secrets externalization, and 50 passing tests with zero lint or type errors.

Two findings require immediate attention before the pipeline is production-ready:

1. **IV-009** — Sync LLM calls inside `async def generate_query_vectors()` will block the event loop, defeating async query processing.
2. **IV-014** — `semantic_vector` is missing from `SELECT_FIELDS`, making the MMR diversity stage (Stage 3) a no-op. All documents appear identical to the diversity algorithm since they all have zero-vector fallbacks.

Both are straightforward fixes (add `semantic_vector` to `SELECT_FIELDS`; switch to `AsyncAzureOpenAI` for LLM calls in `query.py`). After addressing these two critical items and the 7 major findings, the pipeline moves from development-ready to production-ready.
