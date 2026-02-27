<!-- markdownlint-disable-file -->

# AI Search Pipeline — Implementation Validation Log

| Field | Value |
|---|---|
| **Date** | 2026-02-26 |
| **Scope** | Full-quality validation |
| **Files reviewed** | 37 source + 16 test files |
| **Test suite result** | 50 passed, 3 skipped (integration placeholders) |

---

## Summary

| Severity | Count |
|---|---|
| Critical | 1 |
| Major | 7 |
| Minor | 9 |
| **Total** | **17** |

---

## Findings by Category

### Architecture

#### IV-001 — Dead code: extraction sub-modules unused (Minor)

- **Files**: `src/ai_search/extraction/narrative.py`, `src/ai_search/extraction/emotion.py`, `src/ai_search/extraction/objects.py`, `src/ai_search/extraction/low_light.py`
- **Description**: Four extraction sub-modules (`get_narrative`, `get_emotion`, `get_objects`, `get_low_light`) are never imported anywhere in the codebase. The unified `extractor.py` returns a complete `ImageExtraction` model, making these accessor functions dead code.
- **Severity**: Minor
- **Recommendation**: Remove these files or mark as public API utilities and re-export from `extraction/__init__.py`.

#### IV-002 — Dead code: `ingestion/metadata.py` unused (Minor)

- **File**: `src/ai_search/ingestion/metadata.py`
- **Description**: `generate_metadata()` is never imported outside of research docs. The unified `extractor.py` already generates metadata as part of `ImageExtraction`. This module duplicates the LLM call for metadata alone.
- **Severity**: Minor
- **Recommendation**: Remove or mark as deprecated. If kept for standalone metadata generation, document the dual-path design.

#### IV-003 — Module boundary: `ingestion/cli.py` imports directly from `indexing` (Minor)

- **File**: `src/ai_search/ingestion/cli.py`, line 12
- **Description**: The ingestion CLI imports `build_search_document` and `upload_documents` from `ai_search.indexing.indexer`. This creates a cross-layer dependency where the ingestion module reaches into the indexing module. A dedicated orchestrator or pipeline module would better separate concerns.
- **Severity**: Minor
- **Recommendation**: Consider a top-level `pipeline.py` orchestrator or accept this as intentional CLI-level wiring.

---

### Design Principles

#### IV-004 — `lru_cache` on client factories prevents resource cleanup (Major)

- **Files**: `src/ai_search/clients.py` (lines 19, 27, 35), `src/ai_search/config.py` (lines 147, 153, 159, 165)
- **Description**: `@lru_cache(maxsize=1)` is used on all secret loaders and client factories. Cached clients live for the process lifetime with no `close()` or `__del__` hook. For `AsyncAzureOpenAI`, the cached client's `httpx.AsyncClient` transport is never explicitly closed, potentially leaking connections in long-running processes or serverless environments. The `get_search_client` function is notably *not* cached, creating inconsistency.
- **Severity**: Major
- **Recommendation**: Add a `shutdown()` or `close_clients()` function that calls `.close()` on cached clients and clears the caches. Document that these singletons are intended for CLI/script usage.

#### IV-005 — `rerank_candidates` mutates input list in-place (Major)

- **File**: `src/ai_search/retrieval/reranker.py`, lines 102-113
- **Description**: `rerank_candidates()` adds a `rerank_score` key to each dict in the input `candidates` list and then sorts it in-place. Callers cannot preserve the original order. This side-effect is undocumented.
- **Severity**: Major
- **Recommendation**: Either document the mutation explicitly or work on a shallow copy: `scored = [dict(doc, rerank_score=compute_rerank_score(doc, context)) for doc in candidates]`.

#### IV-006 — `embed_text_sync` and `retrieve_sync` use `asyncio.run` — unsafe if event loop already running (Major)

- **Files**: `src/ai_search/embeddings/encoder.py` (line 53), `src/ai_search/retrieval/pipeline.py` (line 125), `src/ai_search/retrieval/query.py` (line 74)
- **Description**: `asyncio.run()` raises `RuntimeError` if called inside an existing event loop (for example, Jupyter notebooks, FastAPI, or any async framework). Three sync wrappers all use this pattern.
- **Severity**: Major
- **Recommendation**: Use `asyncio.get_event_loop().run_until_complete()` with a check, or document that these sync wrappers are CLI-only. For broader compatibility, consider `nest_asyncio` or restructure to avoid the pattern.

---

### DRY Analysis

#### IV-007 — Repetitive thin wrapper pattern in `embeddings/semantic.py`, `structural.py`, `style.py` (Minor)

- **Files**: `src/ai_search/embeddings/semantic.py`, `src/ai_search/embeddings/structural.py`, `src/ai_search/embeddings/style.py`
- **Description**: Each file is a near-identical 13-line wrapper that calls `embed_text(description, dimensions=config.index.vector_dimensions.<field>)`. The only difference is the dimension field name. This could be a single parameterized function.
- **Severity**: Minor
- **Recommendation**: Accept as intentional for clarity and future divergence (each embedding type may grow different preprocessing), or consolidate into a factory function.

#### IV-008 — `SearchDocument` character vector fields are manually expanded (Minor)

- **File**: `src/ai_search/models.py`, lines 99-107
- **Description**: Nine character vector fields (`char_0_semantic_vector` through `char_2_pose_vector`) are manually declared. If `max_character_slots` changes in config, the model must be manually updated to match.
- **Severity**: Minor
- **Recommendation**: Document the coupling between `max_character_slots=3` in config and the 9 hardcoded fields. Consider using `model_validator` to assert consistency.

---

### API and Library Usage

#### IV-009 — `query.py` mixes sync and async OpenAI calls within an async function (Critical)

- **File**: `src/ai_search/retrieval/query.py`, lines 32-56
- **Description**: `generate_query_vectors()` is an `async` function that calls `client.chat.completions.create()` (synchronous `AzureOpenAI`) for two LLM calls before doing `await asyncio.gather(...)` for embeddings. The synchronous LLM calls block the event loop, defeating the purpose of the async function. Each LLM call can take 1-5 seconds, so the two serial blocking calls add 2-10 seconds of event-loop stalling.
- **Severity**: Critical
- **Recommendation**: Use `get_async_openai_client()` and `await client.chat.completions.create(...)` for the LLM calls, or move the sync calls to a thread pool with `asyncio.to_thread()`.

#### IV-010 — `ImageInput.to_openai_image_content` hardcodes `detail: "high"` (Minor)

- **File**: `src/ai_search/ingestion/loader.py`, lines 33-38
- **Description**: The `detail` parameter is hardcoded to `"high"` in the `to_openai_image_content()` method, while the extraction config has an `image_detail` field. The config value is never used.
- **Severity**: Minor
- **Recommendation**: Accept the config `image_detail` parameter and pass it through, or remove `image_detail` from `ExtractionConfig` if always `"high"`.

#### IV-011 — `ImageInput.from_file` hardcodes JPEG MIME type (Minor)

- **File**: `src/ai_search/ingestion/loader.py`, line 37
- **Description**: The data URI uses `data:image/jpeg;base64,...` regardless of the actual file type. PNG, WebP, or other formats will be sent with incorrect MIME type.
- **Severity**: Minor
- **Recommendation**: Detect MIME type from file extension or magic bytes, e.g., `mimetypes.guess_type(path)`.

---

### Error Handling

#### IV-012 — No error handling in extraction pipeline (`extract_image`) (Major)

- **File**: `src/ai_search/extraction/extractor.py`
- **Description**: `extract_image()` makes an OpenAI API call with no try/except. Transient API errors (rate limits, timeouts, service errors) propagate as unhandled `openai.APIError` or `openai.APITimeoutError`. There is no retry logic, unlike the indexer which has exponential backoff.
- **Severity**: Major
- **Recommendation**: Add retry logic with exponential backoff for transient OpenAI errors (429, 500, 503, timeout), consistent with the indexer's retry pattern.

#### IV-013 — No error handling in embedding encoder (`embed_texts`) (Major)

- **File**: `src/ai_search/embeddings/encoder.py`
- **Description**: `embed_texts()` calls the OpenAI embeddings API with no error handling. If a batch fails mid-way through chunked processing, partial results are lost. No retry logic for rate limits.
- **Severity**: Major
- **Recommendation**: Add retry with backoff on 429/503. Consider returning partial results or raising a descriptive error that includes progress state.

#### IV-014 — `_extract_embeddings` silently falls back to zero vectors (Major)

- **File**: `src/ai_search/retrieval/pipeline.py`, lines 100-115
- **Description**: When semantic embeddings are missing from search results (which is the normal case since `semantic_vector` is not in `SELECT_FIELDS`), the function falls back to zero vectors for every document. The MMR algorithm then computes cosine similarity on zero vectors, which are normalized to 1-vectors (due to `norms = np.where(norms == 0, 1, norms)`), making all documents appear identical. This effectively disables diversity selection.
- **Severity**: Major — The MMR stage 3 is functionally broken for the default configuration.
- **Recommendation**: Either include `semantic_vector` in `SELECT_FIELDS` (costs bandwidth and memory), or issue a second targeted query for embeddings of stage 2 results, or use the query embedding as the single "reference" vector and compute per-document pseudo-diversity from the search score distribution.

---

### Test Coverage

#### IV-015 — No unit tests for `retrieval/pipeline.py` or `retrieval/query.py` (Major — test gap, not source defect)

- **Files**: `tests/test_retrieval/`
- **Description**: The three-stage retrieval pipeline (`retrieve()`) and query vector generation (`generate_query_vectors()`) have zero unit tests. These are the primary orchestration functions of the retrieval module. The sync/async interplay in `query.py` (IV-009) is also uncovered.
- **Severity**: Noted as test gap (no IV severity assigned to test gaps, but flagged)
- **Recommendation**: Add tests for `retrieve()` with mocked stages, and for `generate_query_vectors()` with mocked LLM + embedding calls.

---

### Security

#### IV-015 — No input validation on OData filter expressions (Minor)

- **File**: `src/ai_search/retrieval/search.py`, line 36
- **Description**: The `odata_filter` parameter is passed directly to `client.search(filter=...)` without sanitization. While Azure AI Search rejects malformed OData, a defense-in-depth approach would validate the filter before sending.
- **Severity**: Minor — Azure SDK provides server-side validation.
- **Recommendation**: Consider basic format validation or an allowlist of filter patterns if the filter string comes from untrusted user input.

#### IV-016 — No secrets detected in source (Pass)

- **Description**: Grep confirmed no API keys, connection strings, or credentials in source code. `.env` is properly gitignored. `.env.example` contains only placeholder values.
- **Severity**: Pass

---

### General / Operational

#### IV-017 — No `structlog` configuration anywhere (Minor)

- **Files**: Six modules import `structlog.get_logger()`, but no module calls `structlog.configure()`.
- **Description**: Without explicit configuration, structlog falls back to its default dev-friendly renderer. In production, JSON output with timestamps, log level, and correlation IDs is typically required.
- **Severity**: Minor
- **Recommendation**: Add a `logging.py` or configure structlog in `__init__.py` with processors for JSON rendering, timestamps, and log level filtering.

---

## Holistic Assessment

The AI Search Pipeline demonstrates a well-organized, cleanly layered architecture with clear separation between ingestion, extraction, embedding, indexing, and retrieval modules. The codebase follows modern Python practices: Pydantic models with validation constraints, `BaseSettings` for secrets management, structured logging with structlog, async-first embedding generation, and typed function signatures throughout.

**Strengths:**
- Clean module hierarchy with single-responsibility files
- Pydantic models provide strong data contracts with range validation
- Config system cleanly separates secrets (`.env`) from tuning parameters (`config.yaml`)
- Comprehensive index schema with proper HNSW vector search and semantic ranker
- Well-structured unit tests with proper mocking patterns
- Upload retry logic with exponential backoff in the indexer

**Critical issue:**
The single critical finding (IV-009) is that `generate_query_vectors()` blocks the event loop with synchronous OpenAI calls inside an async function. This is a correctness bug that undermines the async design of the retrieval pipeline.

**Major concerns:**
The most impactful major issue is IV-014: the MMR diversity stage is functionally inert because it operates on zero-vector fallbacks (search results don't contain embedding vectors). This means stage 3 of the retrieval pipeline provides no diversity benefit in its current form. Combined with the lack of retry logic in extraction and embedding (IV-012, IV-013), and mutation-based reranking (IV-005), the pipeline has several robustness gaps that should be addressed before production use.

**Overall grade: Solid foundation with targeted fixes needed.** The architecture and code quality are high. Fixing IV-009 (sync-in-async) and IV-014 (zero-vector MMR) would bring the pipeline from "development-ready" to "production-ready." The remaining major items (retry logic, resource cleanup, input mutation) are standard hardening tasks.
