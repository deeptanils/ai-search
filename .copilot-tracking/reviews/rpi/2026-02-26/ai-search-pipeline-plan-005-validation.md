<!-- markdownlint-disable-file -->
# RPI Validation: Phase 5 — Azure AI Search Indexing

**Plan**: `.copilot-tracking/plans/2026-02-26/ai-search-pipeline-plan.instructions.md`
**Changes Log**: `.copilot-tracking/changes/2026-02-26/ai-search-pipeline-changes.md`
**Details Reference**: `.copilot-tracking/details/2026-02-26/ai-search-pipeline-details.md` (Lines 1267–1368)
**Validation Date**: 2026-02-26
**Phase Status**: ✅ PASS

---

## Step-by-Step Validation

### Step 5.1: Create indexing/schema.py — full index definition with HNSW config

**Status**: ✅ PASS

| Criterion | Plan Specification | Implementation Evidence | Result |
|-----------|-------------------|------------------------|--------|
| File exists | `src/ai_search/indexing/schema.py` | File present, 131 lines | ✅ |
| HNSW config | m=4, ef_construction=400, ef_search=500, cosine | `HnswParameters(m=hnsw.m, ef_construction=hnsw.ef_construction, ef_search=hnsw.ef_search, metric="cosine")` | ✅ |
| Shared profile | All vector fields reference `hnsw-cosine-profile` | `PROFILE_NAME = "hnsw-cosine-profile"`; `_build_vector_field()` defaults to this profile | ✅ |
| Semantic ranker | `generation_prompt` as content field | `SemanticField(field_name="generation_prompt")` in `SemanticConfiguration` | ✅ |
| Create/update function | `create_or_update_index()` | Calls `client.create_or_update_index(index)` with structlog | ✅ |
| Index name from config | Configurable index name | `name=idx.name` from `config.index.name` | ✅ |
| Total field count | 15 primitives + 3 vectors + 9 character vectors = 27 | Test `test_field_count` asserts `len(index.fields) == 27` | ✅ |

#### Primitive Fields (15)

| # | Field Name | Type | Attributes | In Plan Spec | In Implementation | Status |
|---|-----------|------|------------|--------------|-------------------|--------|
| 1 | `image_id` | String | key, filterable | ✅ Listed | ✅ | ✅ |
| 2 | `generation_prompt` | String | searchable | ✅ Listed | ✅ | ✅ |
| 3 | `scene_type` | String | filterable, facetable | ✅ Listed (filterable) | ✅ (+facetable) | ✅ |
| 4 | `time_of_day` | String | filterable | ❌ Not listed | ✅ Added | ✅ⁱ |
| 5 | `lighting_condition` | String | filterable, facetable | ✅ Listed (filterable) | ✅ (+facetable) | ✅ |
| 6 | `primary_subject` | String | filterable | ❌ Not listed | ✅ Added | ✅ⁱ |
| 7 | `artistic_style` | String | filterable, facetable | ❌ Not listed | ✅ Added | ✅ⁱ |
| 8 | `tags` | Collection(String) | searchable, filterable, facetable | ✅ Listed | ✅ | ✅ |
| 9 | `narrative_theme` | String | filterable | ❌ Not listed | ✅ Added | ✅ⁱ |
| 10 | `narrative_type` | String | filterable | ✅ Listed | ✅ | ✅ |
| 11 | `emotional_polarity` | Double | filterable, sortable | ✅ Listed | ✅ | ✅ |
| 12 | `low_light_score` | Double | filterable | ✅ Listed | ✅ | ✅ |
| 13 | `character_count` | Int32 | filterable, sortable | ✅ Listed (filterable) | ✅ (+sortable) | ✅ |
| 14 | `metadata_json` | String | — | ✅ Listed | ✅ | ✅ |
| 15 | `extraction_json` | String | — | ✅ Listed | ✅ | ✅ |

ⁱ = Additive enhancement: field exists in `SearchDocument` model (Phase 2, Step 2.2) but not explicitly listed in Step 5.1 details. Inclusion is correct because the schema must match the document model.

#### Primary Vector Fields (3)

| Field Name | Dimensions | Profile | Status |
|-----------|------------|---------|--------|
| `semantic_vector` | 3072 | hnsw-cosine-profile | ✅ |
| `structural_vector` | 1024 | hnsw-cosine-profile | ✅ |
| `style_vector` | 512 | hnsw-cosine-profile | ✅ |

#### Character Vector Fields (9 = 3 slots × 3 types)

| Field Name | Dimensions | Profile | Status |
|-----------|------------|---------|--------|
| `char_0_semantic_vector` | 512 | hnsw-cosine-profile | ✅ |
| `char_0_emotion_vector` | 256 | hnsw-cosine-profile | ✅ |
| `char_0_pose_vector` | 256 | hnsw-cosine-profile | ✅ |
| `char_1_semantic_vector` | 512 | hnsw-cosine-profile | ✅ |
| `char_1_emotion_vector` | 256 | hnsw-cosine-profile | ✅ |
| `char_1_pose_vector` | 256 | hnsw-cosine-profile | ✅ |
| `char_2_semantic_vector` | 512 | hnsw-cosine-profile | ✅ |
| `char_2_emotion_vector` | 256 | hnsw-cosine-profile | ✅ |
| `char_2_pose_vector` | 256 | hnsw-cosine-profile | ✅ |

**HNSW cross-reference with config.yaml**:
- `m: 4` ✅
- `ef_construction: 400` ✅
- `ef_search: 500` ✅
- `max_character_slots: 3` ✅

**Tests**: `tests/test_indexing/test_schema.py` — 4 tests:
1. `test_field_count` — asserts 27 fields ✅
2. `test_key_field` — asserts `image_id` is the single key field ✅
3. `test_vector_field_dimensions` — checks semantic=3072, structural=1024, style=512, char_0_semantic=512, char_0_emotion=256, char_0_pose=256 ✅
4. `test_hnsw_profile` — verifies HNSW algorithm name is `hnsw-cosine` ✅

**Source**: `src/ai_search/indexing/schema.py` — matches plan specification in details (Lines 1267–1293) with additive primitive fields.

---

### Step 5.2: Create indexing/indexer.py — batch document upload with retry

**Status**: ✅ PASS

| Criterion | Plan Specification | Implementation Evidence | Result |
|-----------|-------------------|------------------------|--------|
| File exists | `src/ai_search/indexing/indexer.py` | File present, 107 lines | ✅ |
| Batch splitting | Configurable batch size (default 500) | `batch_size = config.batch.index_batch_size`; config.yaml: `index_batch_size: 500` | ✅ |
| Exponential backoff | Retry on 429/503 | `if e.status_code in (429, 503)` with `delay = base_delay * (2**attempt)` | ✅ |
| Max retries | Configurable retries | `max_retries: int = 3` parameter | ✅ |
| Progress logging | structlog progress | `logger.info("Batch uploaded", ...)` with batch_start, batch_size, succeeded | ✅ |
| Error logging | Log failures | `logger.error("Batch upload failed", ...)` and `logger.warning("Retrying batch upload", ...)` | ✅ |
| `build_search_document()` | Assemble from ImageInput + ImageExtraction + ImageVectors | Function present, flattens character vectors to `char_N_*_vector` fields | ✅ |
| Character flattening | Top-level `char_N_*_vector` fields | Loop `for i, cv in enumerate(vectors.character_vectors[:3])` populating dict | ✅ |
| Empty vector handling | Skip empty vector fields | `_prepare_document()` filters out `k: v` where `isinstance(v, list) and len(v) == 0` | ✅ |
| Return value | Number of successfully uploaded documents | `return total_uploaded` with per-batch counting | ✅ |

**Retry Logic Detail**:
- Catches `HttpResponseError` from `azure.core.exceptions`
- Checks `e.status_code in (429, 503)` — exact match to plan
- Applies `delay = base_delay * (2**attempt)` — exponential backoff
- Uses `time.sleep(delay)` — synchronous retry (batch upload is synchronous)
- Re-raises on non-retryable errors or exhausted retries

**Tests**: `tests/test_indexing/test_indexer.py` — 3 tests:
1. `test_builds_correct_document` — verifies `build_search_document()` produces correct field values and flattened character vectors ✅
2. `test_successful_upload` — verifies upload returns correct count ✅
3. `test_retry_on_429` — verifies retry with `mock_sleep` on HttpResponseError(429) ✅

**Source**: `src/ai_search/indexing/indexer.py` — matches plan specification in details (Lines 1295–1321).

---

### Step 5.3: Create indexing/cli.py — CLI entry point for index management

**Status**: ✅ PASS

| Criterion | Plan Specification | Implementation Evidence | Result |
|-----------|-------------------|------------------------|--------|
| File exists | `src/ai_search/indexing/cli.py` | File present, 31 lines | ✅ |
| `create` subcommand | `ai-search-index create` | `subparsers.add_parser("create", ...)` | ✅ |
| Entry point registered | `pyproject.toml` script entry | `ai-search-index = "ai_search.indexing.cli:main"` at pyproject.toml line 36 | ✅ |
| Calls create_or_update_index | Delegates to schema module | `index = create_or_update_index()` | ✅ |
| Success message | Print confirmation | `print(f"Index '{index.name}' created/updated successfully")` | ✅ |
| Failure exit code | Exit 1 on no command | `sys.exit(1)` in else branch | ✅ |
| `__main__` guard | Direct execution support | `if __name__ == "__main__": main()` | ✅ |
| argparse-based | Standard library CLI | Uses `argparse.ArgumentParser` with subparsers | ✅ |

**Source**: `src/ai_search/indexing/cli.py` — exact character-for-character match to plan specification in details (Lines 1323–1368).

---

## Package Structure

| Artifact | Status |
|----------|--------|
| `src/ai_search/indexing/__init__.py` | ✅ Present — docstring: "Indexing module — Azure AI Search index schema and document upload." |
| `src/ai_search/indexing/schema.py` | ✅ Verified |
| `src/ai_search/indexing/indexer.py` | ✅ Verified |
| `src/ai_search/indexing/cli.py` | ✅ Verified |

---

## Coverage Assessment

| Metric | Value |
|--------|-------|
| Plan steps covered | 3/3 (100%) |
| Files specified | 4 (including `__init__.py`) |
| Files implemented | 4 |
| Tests covering phase | 7 tests across 2 test files |
| Deviations | 0 |
| Missing items | 0 |
| Additive enhancements | 4 additional primitive fields in schema (from SearchDocument model) |

---

## Findings

| # | Severity | Finding | Evidence |
|---|----------|---------|----------|
| F5-01 | INFO | Schema includes 4 additional primitive fields (`time_of_day`, `primary_subject`, `artistic_style`, `narrative_theme`) beyond the 11 listed in the plan's Step 5.1 specification. These are correctly derived from the `SearchDocument` model (Step 2.2) and are additive enhancements. | `schema.py` Lines 80–88; `models.py` `SearchDocument` class |
| F5-02 | INFO | Some fields have additional attributes beyond what the plan specified: `scene_type` and `lighting_condition` gained `facetable`, `artistic_style` gained `facetable`, `character_count` gained `sortable`. All additive, no removals. | `schema.py` Lines 79–97 |
| F5-03 | INFO | `_prepare_document()` helper filters out empty vector lists before upload, preventing Azure AI Search from rejecting documents with zero-length vectors. This is an implementation detail not explicitly in the plan but necessary for correctness. | `indexer.py` Lines 105–107 |
| F5-04 | INFO | `_build_vector_field()` helper function introduced for DRY principle — creates consistent vector field definitions. Not specified in plan but improves maintainability. | `schema.py` Lines 32–42 |

---

## Recommended Next Validations

1. **Phase 6 validation** — Retrieval Service (Steps 6.1–6.5)
2. **Cross-phase integration** — Verify `build_search_document()` correctly maps all 27 fields from extraction + vectors output
3. **Schema version tracking** — Plan references requirements.md Section 12 mandatory rule for index schema versioning; verify implementation or document as follow-on work item
