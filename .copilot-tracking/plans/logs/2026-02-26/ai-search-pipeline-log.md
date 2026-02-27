<!-- markdownlint-disable-file -->
# Planning Log: Candidate Generation & AI Search Pipeline

## Validation Status

**Validated**: 2026-02-26
**Status**: PASS (with findings)
**Planning log path**: `.copilot-tracking/plans/logs/2026-02-26/ai-search-pipeline-log.md`

### Findings

| ID | Severity | Finding | Affected Artifact |
|----|----------|---------|-------------------|
| VF-01 | MAJOR | Plan Phase 8 (Steps 8.1-8.5) line references to details file are out of range. Details file is 1680 lines; plan references Lines 1705-1970. Step 7.2 (Lines 1662-1700) also exceeds by 20 lines. Phase 8 content exists in details (~Lines 1585-1670) but references are off by ~120-290 lines. | Plan → Details line refs |
| VF-02 | MINOR | Planning log IP section covers 4 major alternatives but omits several research alternatives: multiple GPT-4o calls, `azure-ai-inference` SDK (only in DD-01), Azure AI Search semantic ranker only, aggregated/mean-pooled character vectors, separate character index. | Planning log IP section |
| VF-03 | MINOR | DR-04 (Index alias management) is identified as deferred but has no corresponding follow-on WI. Docker containerization and A/B testing framework from research "Potential Next Research" are neither tracked as DRs nor WIs. | Planning log WI section |
| VF-04 | MINOR | No explicit success criterion verifying "Azure AI Foundry only" constraint enforcement (requirement 7). No specific criterion for "no hardcoded secrets" beyond config loading. | Plan Success Criteria |
| VF-05 | MINOR | Requirements v2 Section 12 mandates "Index schema versioned" but the plan has no explicit step or mechanism for schema versioning. DR-04 covers alias management but not the versioning rule itself. | Plan vs Requirements |
| VF-06 | MINOR | Requirements v1 Section 8 ("incremental re-indexing") and Section 9 ("Observability") are not addressed or explicitly deferred. `structlog` is included as a dependency but no observability module is planned. | Plan vs Requirements v1 |

### Clarifying Questions

None. All user requirements are traceable through the plan. The findings above are documentation/accuracy issues, not architectural gaps.

---

## Discrepancy Log

Gaps and differences identified between research findings and the implementation plan.

### Unaddressed Research Items

* DR-01: Azure OpenAI Batch API for 10M+ scale processing
  * Source: azure-ai-foundry-sdk-research.md (Lines 430-580)
  * Reason: Batch API is a future optimization for scale; initial implementation focuses on synchronous + async per-image pipeline. Batch API requires a separate orchestration layer.
  * Impact: Low — does not affect initial MVP functionality

* DR-02: PTU (Provisioned Throughput Units) procurement and capacity planning
  * Source: ai-search-pipeline-research.md (Lines 380-395)
  * Reason: PTU is an operational/procurement decision, not a code implementation task. Infrastructure planning is outside current scope.
  * Impact: Low — affects cost, not functionality

* DR-03: Scalar/binary quantization for vector storage compression
  * Source: ai-search-pipeline-research.md (Lines 47-48)
  * Reason: Optimization for storage at 10M+ scale. Can be enabled via index configuration change without code modifications.
  * Impact: Low — ~50% storage savings deferred to scaling phase

* DR-04: Index alias management for zero-downtime migrations
  * Source: ai-search-pipeline-research.md (Lines 48)
  * Reason: Operational concern for production deployments. Not needed for initial development.
  * Impact: Low — required before production deployment

* DR-05: GPT-4o-mini cost optimization for simpler images
  * Source: ai-search-pipeline-research.md (Lines 50)
  * Reason: Requires image complexity classifier and A/B testing framework. Optimization phase.
  * Impact: Medium — could reduce extraction costs by 30-50% for simple scenes

* DR-06: Evaluation framework (NDCG@K, MRR, precision@K)
  * Source: ai-search-pipeline-research.md (Lines 49)
  * Reason: Requires ground-truth labeled dataset. Research + implementation is a separate planning task.
  * Impact: Medium — needed to validate retrieval quality improvements

* DR-07: Azure Entra ID token-based authentication (DefaultAzureCredential)
  * Source: azure-ai-foundry-sdk-research.md (Lines 60-70)
  * Reason: Plan uses API key auth for development simplicity. Entra ID is a production hardening task.
  * Impact: Low — API key auth is functional; Entra ID is a security improvement

* DR-08: Query expansion via LLM for retrieval improvement
  * Source: hybrid-retrieval-research.md (Lines 500-600)
  * Reason: Adds latency (GPT-4o call in query path). Deferred to optimization phase, can be toggled via config.
  * Impact: Low — retrieval works without expansion; improves recall for ambiguous queries

* DR-09: Caching strategies (embedding cache, result cache)
  * Source: hybrid-retrieval-research.md (Lines 800-850)
  * Reason: Optimization for repeated queries. Not needed for initial implementation.
  * Impact: Low — reduces latency and cost for repeat queries

### Plan Deviations from Research

* DD-01: SDK selection — Plan uses `openai` SDK exclusively, research also documents `azure-ai-inference` as alternative
  * Research recommends: `openai` SDK as primary, `azure-ai-inference` as alternative for non-OpenAI models
  * Plan implements: `openai` SDK only
  * Rationale: Requirements specify Azure AI Foundry with GPT-4o and text-embedding-3-large only. `azure-ai-inference` adds no value for these models and has less mature documentation.

* DD-02: Extraction architecture — Plan uses unified extractor with sub-module re-exports, research confirms single call but requirements list separate module files
  * Research recommends: Single GPT-4o vision call per image
  * Plan implements: Unified extractor (`extraction/extractor.py`) with thin re-export sub-modules (`narrative.py`, `emotion.py`, `objects.py`, `low_light.py`)
  * Rationale: Satisfies both the single-call optimization and the requirements.md Section 11 directory structure. Sub-modules provide typed accessor functions for each dimension.

* DD-03: Embedding grouping — Research suggests 4 parallel embedding calls grouped by dimension, plan groups differently
  * Research recommends: Group by dimension (3072, 1024, 512, 256) for batch efficiency
  * Plan implements: 4 parallel calls but via type-specific wrappers (semantic, structural, style, character) rather than strict dimension grouping
  * Rationale: Character vectors bundle 3 dimensions (512+256+256) in one module for cohesion. The async pipeline still parallelizes all calls via `asyncio.gather`. Net API call count matches research (4 calls).

* DD-04: Re-ranking approach — Research documents both LLM and rule-based; plan selects rule-based only
  * Research recommends: Rule-based for real-time, LLM for offline/premium
  * Plan implements: Rule-based only (Stage 2)
  * Rationale: P95 < 300ms latency constraint eliminates LLM re-ranking in the real-time path. LLM re-ranking documented as follow-on work item (WI-03).

## Implementation Paths Considered

### Selected: Unified GPT-4o + text-embedding-3-large + Flattened Multi-Vector Index

* Approach: Single GPT-4o call extracts all descriptions/metadata → text-embedding-3-large at 4 dimension levels → flattened top-level vector fields in Azure AI Search → three-stage retrieval (RRF → rule-based re-rank → MMR)
* Rationale: Satisfies Azure AI Foundry-only constraint, minimizes API calls (1 LLM + 4 embedding per image), works within Azure AI Search vector field limitations, meets P95 < 300ms retrieval budget
* Evidence: ai-search-pipeline-research.md (Lines 240-290), verified via SDK verification subagent (5/5 checks passed HIGH confidence)

### IP-01: Multi-Model Extraction (DINOv2 + Style Encoder + GPT-4o)

* Approach: Dedicated vision models for structural (DINOv2) and style (Style Encoder) embeddings, GPT-4o for narrative/character
* Trade-offs: Higher structural quality (~15% improvement), higher style quality (~20% improvement), but requires custom model deployment on Azure AI Foundry, 3x more API calls, higher operational complexity
* Rejection rationale: DINOv2 and Style Encoder not available as managed models on Azure AI Foundry. Would require custom container deployment, adding infrastructure complexity. Deferred to WI-06.

### IP-02: Nested Character Vectors (ComplexType Collection)

* Approach: Store character vectors inside `Collection(Edm.ComplexType)` fields for unlimited character support
* Trade-offs: Cleaner schema, unlimited characters, but vector search not supported inside complex types
* Rejection rationale: Azure AI Search explicitly does not support vector search on nested complex type fields. Verified in SDK documentation. Flattened approach selected with 3-slot cap.

### IP-03: LLM Re-Ranking in Real-Time Path

* Approach: Use GPT-4o or Claude for semantic re-ranking of top-50 candidates
* Trade-offs: Higher re-ranking quality, but 500ms+ latency per call, exceeds 300ms budget
* Rejection rationale: Latency budget violation. Rule-based re-ranking achieves sufficient quality for initial release. LLM re-ranking reserved for offline evaluation and future premium tier (WI-03).

### IP-04: Separate Indices Per Vector Type

* Approach: One index per vector dimension (semantic index, structural index, style index) with cross-index merge
* Trade-offs: Simpler per-index schema, but complex cross-index fusion logic, increased search latency, no native RRF
* Rejection rationale: Azure AI Search supports multiple vector fields in a single index with RRF fusion. Single index is simpler and faster.

## Suggested Follow-On Work

Items identified during planning that fall outside current scope.

* WI-01: Azure OpenAI Batch API integration — Implement batch processing pipeline for 10M+ image scale using Azure OpenAI Batch API (50% cost reduction, 24h SLA) (High priority)
  * Source: azure-ai-foundry-sdk-research.md (Lines 430-580), ai-search-pipeline-research.md Key Discovery #9
  * Dependency: Phase 3 (extraction) and Phase 4 (embeddings) must be complete

* WI-02: Evaluation framework — Build NDCG@K, MRR, precision@K, diversity@K metrics with ground-truth labeled dataset (High priority)
  * Source: ai-search-pipeline-research.md Potential Next Research
  * Dependency: Phase 6 (retrieval) must be complete, labeled dataset required

* WI-03: LLM re-ranking for premium/offline tier — Implement GPT-4o-based re-ranking for highest quality results in non-latency-sensitive paths (Medium priority)
  * Source: hybrid-retrieval-research.md (Lines 500-600)
  * Dependency: Phase 6 (retrieval) must be complete

* WI-04: Scalar/binary quantization — Enable vector quantization in Azure AI Search index for ~50% storage reduction (Medium priority)
  * Source: ai-search-pipeline-research.md Key Discovery #3
  * Dependency: Index schema (Phase 5) must be complete, sufficient data to measure quality impact

* WI-05: Azure Entra ID authentication — Replace API key auth with DefaultAzureCredential for production security (Medium priority)
  * Source: azure-ai-foundry-sdk-research.md (Lines 60-70)
  * Dependency: None (can be done independently)

* WI-06: Custom vision model deployment — Deploy DINOv2 and/or Style Encoder as custom models on Azure AI Foundry for improved structural/style quality (Low priority)
  * Source: multi-vector-encoding-research.md (Lines 140-200)
  * Dependency: Azure AI Foundry custom model container infrastructure

* WI-07: Query expansion via LLM — Pre-expand ambiguous queries using GPT-4o before search (Low priority)
  * Source: hybrid-retrieval-research.md (Lines 500-600)
  * Dependency: Phase 6 (retrieval) must be complete

* WI-08: Embedding and result caching — Implement Redis or in-memory caching for repeated query embeddings and search results (Low priority)
  * Source: hybrid-retrieval-research.md (Lines 800-850)
  * Dependency: Phase 6 (retrieval) must be complete

* WI-09: CI/CD pipeline with UV — GitHub Actions for testing, linting, type checking, and Docker image builds (Medium priority)
  * Source: ai-search-pipeline-research.md Potential Next Research
  * Dependency: Phase 8 (tests) and Phase 9 (validation) must be complete

* WI-10: GPT-4o-mini cost optimization — Route simpler images to GPT-4o-mini for extraction cost reduction (Low priority)
  * Source: ai-search-pipeline-research.md Potential Next Research
  * Dependency: Phase 3 (extraction) must be complete, image complexity classifier needed

## Implementation Findings

Findings discovered during implementation (Phases 1-9).

### Runtime Deviations

* ID-01: OpenAI SDK type stubs do not support multimodal content lists in message dicts
  * Files affected: `extraction/extractor.py`, `ingestion/metadata.py`
  * Resolution: `type: ignore[misc, list-item]` comments added — no runtime impact

* ID-02: Azure Search SDK types `VectorizedQuery` subclass not recognized by `vector_queries` parameter type
  * Files affected: `retrieval/search.py`
  * Resolution: `type: ignore[arg-type]` comment added — no runtime impact

* ID-03: `SearchableField` and `SimpleField` are factory functions, not types
  * Files affected: `indexing/schema.py`
  * Resolution: Changed field list type annotation to `list[SearchField]` — all return `SearchField` at runtime

### Implementation Follow-On Work

* WI-11: Add structured logging configuration — Configure structlog formatters, processors, and log levels (Medium priority)
  * Source: Phase 7, CLI entry points
  * Dependency: None

* WI-12: Add image query support to retrieval CLI — Currently text-only; image queries require extraction pipeline integration (Medium priority)
  * Source: Phase 6, Step 6.1
  * Dependency: Phases 3-4 complete

* WI-13: Add batch ingestion CLI — Process multiple images from JSON manifest (Low priority)
  * Source: Phase 7, Step 7.1
  * Dependency: Phase 7 complete

* WI-14: Implement index versioning — Track schema versions per requirements.md Section 12 (Low priority)
  * Source: VF-05 validation finding
  * Dependency: Phase 5 complete

* WI-15: Implement end-to-end integration tests — Fill placeholder tests with real Azure calls (Medium priority)
  * Source: Phase 8, Step 8.5
  * Dependency: Azure credentials configured

## User Decisions

No user decisions were required during implementation. All ambiguities were resolved using research findings and plan specifications.
