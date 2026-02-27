<!-- markdownlint-disable-file -->
# RPI Validation: Phase 7, CLI Entry Points

## Validation Summary

| Field | Value |
|-------|-------|
| Plan | ai-search-pipeline-plan.instructions.md |
| Phase | 7 (CLI Entry Points) |
| Steps Validated | 7.1, 7.2 |
| Status | PASS |
| Validation Date | 2026-02-26 |

## Step-by-Step Validation

### Step 7.1: Create ingestion/cli.py, ingest command (ai-search-ingest)

| Criterion | Status | Evidence |
|-----------|--------|----------|
| File exists at `src/ai_search/ingestion/cli.py` | PASS | File present, 63 lines |
| Accepts `--image-url` argument | PASS | `parser.add_argument("--image-url", ...)` |
| Accepts `--image-file` argument | PASS | `parser.add_argument("--image-file", ...)` |
| Accepts `--prompt` argument (required) | PASS | `parser.add_argument("--prompt", ..., required=True)` |
| Accepts `--image-id` argument (required) | PASS | `parser.add_argument("--image-id", ..., required=True)` |
| Requires either `--image-url` or `--image-file` | PASS | Validation at lines 52-54 with `sys.exit(1)` on missing |
| Runs extraction pipeline | PASS | Calls `extract_image(image_input)` |
| Runs embedding pipeline | PASS | Calls `await generate_all_vectors(extraction)` |
| Runs index upload | PASS | Calls `upload_documents([doc])` via `build_search_document()` |
| Prints summary | PASS | `print(f"Successfully indexed image '{image_input.image_id}'")` |
| Registered as `ai-search-ingest` entry point | PASS | `pyproject.toml` line 37: `ai-search-ingest = "ai_search.ingestion.cli:main"` |
| Uses `ImageInput.from_url` / `ImageInput.from_file` | PASS | Conditional construction based on `args.image_url` |
| Async pipeline with `asyncio.run` | PASS | `asyncio.run(_process_image(image_input))` in `main()` |
| Structured logging throughout | PASS | `structlog.get_logger()` with info-level messages at each stage |
| Changes log entry matches | PASS | "Ingestion CLI entry point (ai-search-ingest)" |

Plan-to-implementation alignment: Full match. All success criteria from the plan met, including the entry point registration.

### Step 7.2: Create retrieval/cli.py, query command (ai-search-query)

| Criterion | Status | Evidence |
|-----------|--------|----------|
| File exists at `src/ai_search/retrieval/cli.py` | PASS | File present, 49 lines |
| Accepts `--query TEXT` (required) | PASS | `parser.add_argument("--query", ..., required=True)` |
| Accepts `--top N` with default 10 | PASS | `parser.add_argument("--top", type=int, default=10)` |
| Accepts `--filter ODATA` | PASS | `parser.add_argument("--filter", ..., default=None)` |
| Prints ranked results with scores | PASS | Formatted output with search_score, rerank_score, scene_type, prompt, tags |
| Calls retrieval pipeline | PASS | `retrieve_sync(args.query, context=context, odata_filter=args.filter, top=args.top)` |
| Builds `QueryContext` from query text | PASS | `QueryContext(query_text=args.query)` |
| Handles empty results | PASS | Prints "No results found." and returns |
| Registered as `ai-search-query` entry point | PASS | `pyproject.toml` line 39: `ai-search-query = "ai_search.retrieval.cli:main"` |
| Result display includes image_id, scores, scene_type, prompt, tags | PASS | Lines 30-47: formatted output with truncated prompt (80 chars) |
| Changes log entry matches | PASS | "Query CLI entry point (ai-search-query)" |

Plan-to-implementation alignment: Full match. All success criteria from the plan met.

## Cross-Reference: pyproject.toml Entry Points

| Entry Point | Module | Plan Reference | Registered |
|-------------|--------|----------------|------------|
| `ai-search-ingest` | `ai_search.ingestion.cli:main` | Step 7.1 | Yes |
| `ai-search-query` | `ai_search.retrieval.cli:main` | Step 7.2 | Yes |
| `ai-search-index` | `ai_search.indexing.cli:main` | Step 5.3 (Phase 5) | Yes |

All three CLI entry points are registered in `pyproject.toml` under `[project.scripts]`.

## Coverage Assessment

### Plan Coverage

| Step | Planned | Implemented | Changes Logged | Entry Point Registered |
|------|---------|-------------|----------------|----------------------|
| 7.1 | ingestion/cli.py | Yes | Yes | Yes (`ai-search-ingest`) |
| 7.2 | retrieval/cli.py | Yes | Yes | Yes (`ai-search-query`) |

Both planned files exist and match their specifications.

### Deviations

No deviations detected. Both CLI modules implement exactly what the plan specified.

### Test Coverage Observations

| Observation | Severity | Notes |
|-------------|----------|-------|
| No dedicated unit tests for CLI modules | Low | CLI modules are thin wrappers over the pipeline functions. Testing would involve argparse and full pipeline mocking. Integration testing via `test_end_to_end.py` is more appropriate. |

## Findings

| # | Finding | Severity | Category |
|---|---------|----------|----------|
| 1 | Both CLI entry points implement their planned specifications completely | Info | Coverage |
| 2 | All three CLI entry points (ingest, index, query) registered in pyproject.toml | Info | Configuration |
| 3 | Ingestion CLI runs full pipeline (extraction → embedding → indexing) in async context | Info | Correctness |
| 4 | Query CLI uses sync wrapper `retrieve_sync()` to avoid requiring async in argparse flow | Info | Design |
| 5 | Query CLI result display truncates long prompts at 80 characters | Info | UX |
| 6 | Ingestion CLI validates mutual exclusivity of `--image-url` and `--image-file` at runtime (not argparse mutually exclusive group) | Low | Usability |
| 7 | Neither CLI module catches or handles exceptions from Azure service calls | Low | Robustness |

### Finding 6 Detail

The plan specifies `--image-url` or `--image-file`. The implementation uses manual validation rather than `argparse.add_mutually_exclusive_group()`. Both approaches satisfy the requirement, though the argparse group would provide better help text. This is a minor style preference, not a defect.

### Finding 7 Detail

Both CLI modules allow exceptions to propagate unhandled. For a production CLI, wrapping in try/except with user-friendly error messages would improve the experience. This is acceptable for an initial implementation and would be addressed in hardening work.

## Phase Status

PASS: All 2 steps fully implemented, changes log accurately reflects implementation, no deviations detected.

## Recommended Next Validations

1. Phase 8 (Tests) validation to confirm test coverage across all modules
2. End-to-end manual testing of CLI commands against a live Azure environment
3. Error handling and edge-case hardening pass for both CLI modules
