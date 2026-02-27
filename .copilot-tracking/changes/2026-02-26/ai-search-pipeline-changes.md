<!-- markdownlint-disable-file -->
# Release Changes: Candidate Generation & AI Search Pipeline

**Related Plan**: ai-search-pipeline-plan.instructions.md
**Implementation Date**: 2026-02-26

## Summary

Full implementation of the Candidate Generation & AI Search Pipeline: project scaffolding with UV, configuration management, GPT-4o vision extraction, multi-vector Matryoshka embeddings, Azure AI Search indexing with flattened character vectors, three-stage hybrid retrieval (RRF → re-rank → MMR), CLI entry points, and unit tests.

## Changes

### Added

* `pyproject.toml` — Project configuration: hatchling build, 11 runtime deps, 7 dev deps, 3 CLI entry points, ruff/mypy/pytest configuration
* `.python-version` — Pins Python 3.12
* `config.yaml` — Non-secret configuration (models, weights, dimensions, HNSW params, retrieval settings, batch sizes)
* `.env.example` — Azure credential variable templates
* `.gitignore` — Comprehensive ignore patterns for Python, IDE, env files
* `README.md` — Setup/usage/development documentation
* `src/ai_search/__init__.py` — Package root with `__version__ = "0.1.0"`
* `src/ai_search/ingestion/__init__.py` — Ingestion subpackage
* `src/ai_search/extraction/__init__.py` — Extraction subpackage
* `src/ai_search/embeddings/__init__.py` — Embeddings subpackage
* `src/ai_search/indexing/__init__.py` — Indexing subpackage
* `src/ai_search/retrieval/__init__.py` — Retrieval subpackage
* `src/ai_search/config.py` — Configuration management (pydantic-settings for .env, PyYAML for config.yaml, @lru_cache loaders)
* `src/ai_search/models.py` — Shared Pydantic models (ImageExtraction, SearchDocument, QueryContext, SearchResult, etc.)
* `src/ai_search/clients.py` — Client factories for AzureOpenAI, AsyncAzureOpenAI, SearchIndexClient, SearchClient
* `src/ai_search/ingestion/loader.py` — ImageInput model with from_url/from_file/to_openai_image_content
* `src/ai_search/ingestion/metadata.py` — Synthetic metadata generation via GPT-4o structured output
* `src/ai_search/ingestion/cli.py` — Ingestion CLI entry point (ai-search-ingest)
* `src/ai_search/extraction/extractor.py` — Unified GPT-4o vision extraction with EXTRACTION_SYSTEM_PROMPT
* `src/ai_search/extraction/narrative.py` — Narrative intent accessor from unified extraction
* `src/ai_search/extraction/emotion.py` — Emotional trajectory accessor from unified extraction
* `src/ai_search/extraction/objects.py` — Required objects accessor from unified extraction
* `src/ai_search/extraction/low_light.py` — Low-light metrics accessor from unified extraction
* `src/ai_search/embeddings/encoder.py` — Base embedding encoder (text-embedding-3-large, async, Matryoshka dimensions, batch chunking)
* `src/ai_search/embeddings/semantic.py` — Semantic vector wrapper (3072 dims)
* `src/ai_search/embeddings/structural.py` — Structural vector wrapper (1024 dims)
* `src/ai_search/embeddings/style.py` — Style vector wrapper (512 dims)
* `src/ai_search/embeddings/character.py` — Character sub-vector generation (3 slots × 3 types, batch grouped by dimension)
* `src/ai_search/embeddings/pipeline.py` — Embedding pipeline orchestrator (asyncio.gather parallel execution)
* `src/ai_search/indexing/schema.py` — Full index schema (15 primitives + 3 primary vectors + 9 character vectors, HNSW cosine, semantic ranker)
* `src/ai_search/indexing/indexer.py` — Batch document uploader with exponential backoff retry on 429/503
* `src/ai_search/indexing/cli.py` — Index management CLI (ai-search-index create)
* `src/ai_search/retrieval/query.py` — Query embedding generation (LLM structural/style descriptions + parallel embedding)
* `src/ai_search/retrieval/search.py` — Hybrid search with weighted multi-vector queries (config weights × 10)
* `src/ai_search/retrieval/reranker.py` — Stage 2 rule-based re-ranking (emotional, narrative, object overlap, low-light scoring)
* `src/ai_search/retrieval/diversity.py` — Stage 3 MMR diversity selection with configurable lambda
* `src/ai_search/retrieval/pipeline.py` — Three-stage retrieval orchestrator (RRF → re-rank → MMR)
* `src/ai_search/retrieval/cli.py` — Query CLI entry point (ai-search-query)
* `tests/conftest.py` — Shared fixtures (sample_config, mock_openai_client, sample_image_input, sample_extraction, sample_vectors)
* `tests/test_config.py` — Config loading and default validation tests (6 tests)
* `tests/test_models.py` — Pydantic model validation tests (10 tests)
* `tests/test_extraction/__init__.py` — Test package init
* `tests/test_extraction/test_extractor.py` — Extraction function tests with mocked GPT-4o (2 tests)
* `tests/test_embeddings/__init__.py` — Test package init
* `tests/test_embeddings/test_encoder.py` — Encoder batch chunking and dimension tests (4 tests)
* `tests/test_embeddings/test_pipeline.py` — Pipeline parallel execution test (1 test)
* `tests/test_indexing/__init__.py` — Test package init
* `tests/test_indexing/test_schema.py` — Index schema validation tests (4 tests)
* `tests/test_indexing/test_indexer.py` — Batch upload and retry tests (3 tests)
* `tests/test_retrieval/__init__.py` — Test package init
* `tests/test_retrieval/test_search.py` — Hybrid search query construction test (1 test)
* `tests/test_retrieval/test_reranker.py` — Re-ranking scoring function tests (12 tests)
* `tests/test_retrieval/test_diversity.py` — MMR selection tests (4 tests)
* `tests/test_integration/__init__.py` — Integration test package init
* `tests/test_integration/conftest.py` — Integration-specific fixtures with Azure skip logic
* `tests/test_integration/test_end_to_end.py` — E2E test placeholders with @pytest.mark.integration

### Modified

### Removed

* `main.py` — UV default template file (replaced by CLI entry points)

## Additional or Deviating Changes

* OpenAI message typing: Added `type: ignore[misc, list-item]` comments on multimodal user message dicts in `extractor.py` and `metadata.py` because mypy cannot infer the correct ChatCompletionMessageParam TypedDict from dict literals containing mixed-type `content` lists.
  * This is a known limitation of the OpenAI SDK's type stubs with multimodal content.
* Azure Search `VectorizedQuery` vs `VectorQuery`: Added `type: ignore[arg-type]` on `vector_queries` parameter in `search.py` because the SDK's type stub declares `list[VectorQuery]` but `VectorizedQuery` (a subclass) is the correct runtime type.
  * This is a known SDK typing inconsistency.
* Extraction sub-modules (`narrative.py`, `emotion.py`, `objects.py`, `low_light.py`) implemented as thin typed accessors rather than independent extraction calls, per DD-02 deviation (unified single call vs. separate extraction modules).

## Release Summary

Total files: 53 (37 source + 16 test files)

**Files Created**: 53
* Source (37): pyproject.toml, .python-version, config.yaml, .env.example, .gitignore, README.md, 6 __init__.py, config.py, models.py, clients.py, 2 ingestion modules, 5 extraction modules, 6 embedding modules, 3 indexing modules, 6 retrieval modules
* Tests (16): conftest.py, test_config.py, test_models.py, 2 extraction tests, 3 embedding tests, 3 indexing tests, 4 retrieval tests, 3 integration placeholders

**Files Modified**: 0 (all newly created)

**Files Removed**: 1 (main.py — UV default template)

**Validation Results**:
* `uv run ruff check src/ tests/` — All checks passed
* `uv run mypy src/` — Success: no issues found in 32 source files
* `uv run pytest tests/ -m "not integration"` — 50 passed, 3 deselected (integration), 2 warnings (Azure SDK internal)

**Dependencies**: 52 packages installed via `uv sync`

**Deployment Notes**: Requires `.env` file with Azure AI Foundry and Azure AI Search credentials before running CLI commands.
