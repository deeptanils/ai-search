<!-- markdownlint-disable-file -->
# Planning Log: Multimodal Image Embeddings & Index Enhancements

## Discrepancy Log

Gaps and differences identified between research findings and the implementation plan.

### Unaddressed Research Items

* DR-01: Cohere Embed v4 full integration
  * Source: model-strategy-index-config-research.md (Lines 160-230)
  * Reason: User selected Florence as primary with Cohere as fallback; implementation plan covers Florence only. Cohere is documented as follow-on work (WI-01).
  * Impact: low — Cohere is a future fallback, not required for initial delivery

* DR-02: Cohere `azure-ai-inference` SDK dependency
  * Source: model-strategy-index-config-research.md (Lines 195-215)
  * Reason: Since Cohere is deferred to follow-on work, the `azure-ai-inference` SDK is not added to dependencies.
  * Impact: low — deferred to WI-01

* DR-03: `azure-ai-vision-imageanalysis` SDK option
  * Source: model-strategy-index-config-research.md (Lines 110-115)
  * Reason: Plan uses `httpx` REST API directly instead of the official SDK. Research notes the SDK primarily focuses on image analysis (captions, tags) rather than vectorization endpoints.
  * Impact: low — REST is the recommended approach per research

* DR-04: Florence region availability
  * Source: model-strategy-index-config-research.md (Lines 85-87)
  * Reason: Plan does not include region validation logic. User is expected to provision the CV resource in a supported region.
  * Impact: medium — deployment failure if user provisions in unsupported region

* DR-05: Cost analysis for 10M images
  * Source: model-strategy-index-config-research.md (Lines 410-425)
  * Reason: Cost estimates (~$10K for Florence at 10M images) are documented in research but not included in the plan. Operational concern, not implementation.
  * Impact: low — informational

* DR-06: Scalar quantization for `image_vector` field
  * Source: model-strategy-index-config-research.md (Lines 405-408)
  * Reason: Research mentions Florence adds +37GB at 10M docs. Scalar quantization could reduce this. Deferred to follow-on work (WI-02).
  * Impact: low — optimization, not functional requirement

### Plan Deviations from Research

* DD-01: Florence not in Azure AI Foundry catalog
  * Research recommends: Acknowledges Florence is a separate Azure resource, not in Foundry model catalog
  * Plan implements: Uses Florence as the primary image embedding model despite being outside Foundry catalog
  * Rationale: User explicitly selected Florence as primary model. User's constraint is "Azure cloud services" (broader), not "Foundry catalog only." The user said they will "enable this inside AI Foundry" — meaning they'll provision the Azure services needed.

* DD-02: REST API (`httpx`) instead of `azure-ai-vision-imageanalysis` SDK
  * Research recommends: Lists both options — REST API (recommended) and SDK
  * Plan implements: REST API via `httpx` (already a project dependency)
  * Rationale: Research itself recommends REST for vectorization endpoints. SDK focuses on image analysis features. `httpx` is already in `pyproject.toml` dependencies — no new dependency needed.

* DD-03: Weight redistribution
  * Research recommends: Does not specify exact weight values for the new `image_weight`
  * Plan implements: `image_weight: 0.2`, with `semantic_weight` reduced from 0.4 (assumed previous value) and other weights adjusted so total = 1.0
  * Rationale: Image similarity should have meaningful presence in hybrid search results. 0.2 is a reasonable starting point; weights are configurable via `config.yaml`.

## Implementation Paths Considered

### Selected: Option D — Hybrid text-embedding-3-large + Azure Computer Vision 4.0 (Florence)

* Approach: Keep existing text-embedding-3-large for all text embeddings. Add Florence as a parallel image-to-vector embedding path. Two embedding spaces: text-embedding-3-large space (multi-dim) + Florence space (1024-dim shared image-text).
* Rationale: Preserves all existing multi-vector richness (3072/1024/512/256 dims). Adds direct visual similarity with zero latency impact (fully parallelizable with GPT-4o extraction). Minimal code changes (~1 new file, 4 modified files). Both models are GA, Microsoft-managed.
* Evidence: model-strategy-index-config-research.md (Lines 260-310)

### IP-01: Option A — Hybrid text-embedding-3-large + Florence (identical to Selected)

* Approach: Same as selected Option D
* Trade-offs: N/A — same approach, Option D is the named variant with clearer justification
* Rejection rationale: Merged with Option D

### IP-02: Option B — Replace text-embedding-3-large with Cohere Embed v4

* Approach: Single multimodal model (Cohere Embed v4) for ALL embeddings — both text and image
* Trade-offs: Single model simplifies architecture and stays within Foundry catalog. However, max 1024 dims loses the 3072-dim semantic vector. Different SDK (`azure-ai-inference`). Replaces a proven, battle-tested model. No Matryoshka support equivalent.
* Rejection rationale: Unacceptable loss of 3072-dim semantic vector quality. Too disruptive to existing architecture. User confirmed satisfaction with text-embedding-3-large for text embeddings.

### IP-03: Option C — Keep text-embedding-3-large + Add Cohere Embed v4 for images only

* Approach: Two models — text-embedding-3-large for text, Cohere Embed v4 for image embeddings only
* Trade-offs: Stays within Azure AI Foundry. Adds native multimodal support with configurable dimensions (256-1024). However, different embedding space from text-embedding-3-large, requires separate client and SDK dependency, pay-per-token cost model.
* Rejection rationale: User selected Florence as primary. Cohere documented as fallback (WI-01). Florence is purpose-built for vision with a simpler REST API. Adding `azure-ai-inference` or `cohere` SDK adds dependency complexity that is unnecessary when Florence meets all requirements.

## Suggested Follow-On Work

Items identified during planning that fall outside current scope.

* WI-01: Cohere Embed v4 fallback integration — Implement Cohere Embed v4 as an alternative image embedding provider, selectable via `image_embedding_model` config (medium priority)
  * Source: User selected "Both (Florence primary, Cohere fallback)" and research doc (Lines 160-230)
  * Dependency: Current plan must be completed first (Florence integration)

* WI-02: Scalar quantization for image_vector — Enable scalar or binary quantization on the `image_vector` field to reduce storage by 4-8x at scale (low priority)
  * Source: Research doc (Lines 410-425), storage impact analysis
  * Dependency: Phase 3 completion (field must exist first)

* WI-03: Image-to-image search CLI command — Add a CLI command that accepts an image URL/path and returns visually similar images (medium priority)
  * Source: Research doc (Lines 27-30), user requirement for image similarity search
  * Dependency: Phases 2 + 4 completion (image embedding + retrieval)

* WI-04: Address review rework items — Fix the 1 Critical (IV-009: sync-in-async in query.py), 7 Major, and 9 Minor findings from the implementation review (high priority)
  * Source: `.copilot-tracking/reviews/2026-02-26/ai-search-pipeline-plan-review.md`
  * Dependency: None — can be done in parallel with or before this plan

* WI-05: Integration tests with live Azure services — Add integration tests that call real Florence/OpenAI/Search endpoints with test data (medium priority)
  * Source: Derived from plan Phase 6 (tests are unit-only with mocks)
  * Dependency: Phase 7 completion + Azure resource provisioning

* WI-06: Florence rate limiting and retry logic — Add exponential backoff and rate-limit handling to the Florence client for production workloads (low priority)
  * Source: Research doc (Lines 87-88), Florence rate limits (1000 req/min)
  * Dependency: Phase 2 completion (client module)

* WI-07: Fix `asyncio.run()` sync-wrapper anti-pattern — The plan replicates the known IV-009 anti-pattern (`asyncio.run()` inside sync wrapper). Should use `loop.run_until_complete()` or remove sync wrappers entirely (high priority)
  * Source: Validation finding MN-03, prior review IV-009
  * Dependency: None — can be fixed independently

* WI-08: `lru_cache` on `httpx.AsyncClient` graceful shutdown — Cached async clients prevent clean shutdown. Consider `contextvar` or explicit lifecycle management (low priority)
  * Source: Validation finding MN-04
  * Dependency: Phase 2 completion

## Implementation Deviations

Deviations discovered during implementation execution (Phase 2 execution).

* DD-04: httpx dependency already present in pyproject.toml
  * Plan specifies: Step 5.1 — verify and add `httpx>=0.27` to pyproject.toml
  * Implementation differs: `httpx>=0.27` was already present; no modification needed
  * Rationale: Verified via `grep httpx pyproject.toml`

* DD-05: Two existing tests required fixes not anticipated in plan
  * Plan specifies: Phase 6 — only new test additions
  * Implementation differs: `test_search_weights_sum_to_one` (test_config.py) and `test_field_count` (test_schema.py) failed after implementation; fixed in Phase 7
  * Rationale: Existing tests validated invariants (weights sum to 1.0, field count = 27) that changed with the new image_weight and image_vector field

## Validation History

### Round 1

* Status: NEEDS_REWORK
* Findings: 3 Major, 4 Minor
* MJ-01 (S2 tier undocumented): Fixed — added Service Tier section to plan, S2 guidance to Step 5.2
* MJ-02 (line references off): Fixed — recounted all lines, updated plan references to match actual details file headings
* MJ-03 (Foundry-only waiver): Fixed — added explicit Foundry-only waiver note to plan Standards References section, referencing DD-01
* MN-01 (Cohere docs vague): Accepted — Cohere is deferred to WI-01, documented in README as alternative
* MN-02 (weight rationale): Accepted — DD-03 in planning log documents the rationale
* MN-03 (asyncio.run anti-pattern): Deferred to WI-07 — out of scope for this plan
* MN-04 (lru_cache shutdown): Deferred to WI-08 — out of scope for this plan

---

## Plan Validation Report

**Validation date**: 2026-02-26
**Validation status**: NEEDS_REWORK
**Validated artifacts**:

* Research: `.copilot-tracking/research/2026-02-26/model-strategy-index-config-research.md`
* Plan: `.copilot-tracking/plans/2026-02-26/multimodal-embeddings-plan.instructions.md`
* Details: `.copilot-tracking/details/2026-02-26/multimodal-embeddings-details.md`
* Log: `.copilot-tracking/plans/logs/2026-02-26/multimodal-embeddings-log.md`

### Findings (severity-ordered)

#### Major Findings

* **MJ-01: S2 tier requirement not implemented**
  * Severity: Major
  * Description: User requirement #5 specifies "Target S2 standard tier." The plan's overview text references "update configuration for S2 tier planning" (plan Line 9), but no checklist item, detail step, or success criterion addresses S2 tier configuration, documentation, or validation. The requirement is acknowledged in the overview but never followed through.
  * Recommended fix: Add a step (Phase 1 or Phase 5) that documents S2 tier recommendation in `README.md`, validates storage projections fit within S2 limits (1 TB), and optionally adds a `tier` field to config or index config for operational reference.

* **MJ-02: Line number cross-references off by ~100-130 lines for Steps 2.3 onward**
  * Severity: Major
  * Description: The plan's detail-line references are accurate for Steps 1.1 through 2.1 (Lines 11-172). Starting at Step 2.2, the plan underestimates the line range (says Lines 174-240 but the step's image.py module extends to approximately line 330). All subsequent line references inherit this offset and point to incorrect content. Verified examples:
    * Step 2.3: Plan says Lines 242-282 → Actual line 242 is inside `embed_image()` function (Step 2.2 content). Step 2.3 starts at approximately line 340.
    * Step 3.1: Plan says Lines 288-322 → Actual line 288 is inside `embed_text_for_image_search()` (Step 2.2 content). Step 3.1 starts at approximately line 395.
    * Step 4.1: Plan says Lines 395-448 → Actual line 395 is Step 3.1 content. Step 4.1 starts at approximately line 510.
  * Recommended fix: Re-count line numbers in the details file and update all cross-references from Step 2.2 onward. The details file is 780 lines total; the current references top out at line 665, confirming a systematic undercount.

* **MJ-03: requirements.md Section 12 conflict with Florence model selection**
  * Severity: Major
  * Description: Section 12 of `requirements.md` states the mandatory rule "Azure Foundry only for models." Azure Computer Vision 4.0 (Florence) is NOT an Azure AI Foundry model — it is a separate Azure Cognitive Services resource. The plan acknowledges this in DD-01 (log) and argues the user's intent is broader ("Azure cloud services"), but the mandatory rule as written contradicts the primary model choice. If another team member or reviewer enforces Section 12 literally, this blocks implementation.
  * Recommended fix: Either (a) update `requirements.md` Section 12 to say "Azure AI Foundry or Azure Cognitive Services" to formally permit Florence, or (b) add an explicit waiver note in the plan referencing user confirmation that Florence (outside Foundry catalog) is acceptable. Without one of these, the contradiction remains.

#### Minor Findings

* **MN-01: Cohere Embed v4 fallback documentation is implicit**
  * Severity: Minor
  * Description: User requirement #3 says "Document Cohere Embed v4 as fallback." The plan's Step 5.2 (README update) includes a bullet "That Cohere Embed v4 is the documented alternative (reference research doc)" and the log lists WI-01 for future Cohere integration. However, no detail step provides specific Cohere documentation content — the README step is vague about what to write. The research document has thorough Cohere coverage that could be summarized.
  * Recommended fix: Add concrete Cohere documentation content to Step 5.2 details — at minimum a paragraph covering: SDK (`azure-ai-inference`), endpoint pattern, dimension range (256-1024), and a reference to the research document for full setup instructions.

* **MN-02: Weight redistribution rationale not documented in details**
  * Severity: Minor
  * Description: Step 1.1 reduces `semantic_weight` from 0.5 to 0.4, `structural_weight` from 0.2 to 0.15, and `style_weight` from 0.2 to 0.15 to make room for `image_weight: 0.2`. The details file says "Weights redistributed to sum to 1.0" but does not state the original values or explain why semantic absorbed the largest reduction (-0.1 vs -0.05 for the others). The log entry DD-03 notes the redistribution but calls it a "reasonable starting point."
  * Recommended fix: Add a brief rationale in the Step 1.1 details noting the original weights (0.5/0.2/0.2/0.1) and explaining the redistribution strategy (e.g., "semantic reduced most because it already has the highest weight and the image vector provides complementary visual signal").

* **MN-03: `generate_query_vectors_sync()` uses `asyncio.run()` (sync-in-async concern)**
  * Severity: Minor
  * Description: Step 4.1 details show `generate_query_vectors_sync()` calling `asyncio.run()`. If this function is ever called from within an already-running event loop, it will raise `RuntimeError`. The planning log WI-04 already flags the same pattern (IV-009) as a Critical finding from a prior review for the existing query.py — this step replicates the anti-pattern rather than fixing it.
  * Recommended fix: Either fix IV-009 before or during this implementation (as WI-04 suggests), or add a note that synchronous wrappers should use `asyncio.get_event_loop().run_until_complete()` with a loop check, or remove the sync wrapper entirely if all callers are async.

* **MN-04: `lru_cache` on `httpx.AsyncClient` prevents graceful shutdown**
  * Severity: Minor
  * Description: Step 2.1 uses `@lru_cache(maxsize=1)` on `get_cv_client()` returning `httpx.AsyncClient`. The cached client is never explicitly closed, which can leak connections. This mirrors the existing pattern in `clients.py` for OpenAI clients, so it is consistent with current code, but it is worth noting for production readiness.
  * Recommended fix: No immediate action needed (consistent with existing patterns). Consider adding a cleanup function or context manager for future production hardening (could be tracked as follow-on work).

### Requirement Coverage Matrix

| # | User Requirement | Plan Coverage | Status |
|---|-----------------|---------------|--------|
| 1 | Direct image embeddings (image → vector) | Phase 2 (Steps 2.1-2.3), Phase 3 (Step 3.1) | Covered |
| 2 | Azure Computer Vision 4.0 (Florence) as primary | Throughout plan — Florence is the primary model | Covered |
| 3 | Document Cohere Embed v4 as fallback | Step 5.2 (implicit), WI-01 (log) | Partially covered (MN-01) |
| 4 | Scoring profiles for text boost | Step 3.2 (text-boost profile with generation_prompt 3x, tags 2x) | Covered |
| 5 | Target S2 standard tier | Plan overview mentions it; no implementation step | NOT covered (MJ-01) |
| 6 | requirements.md Section 12 compliance | .env for secrets, config.yaml for config | Covered (but MJ-03 conflict on Foundry-only rule) |

### Success Criteria Traceability

All 9 success criteria in the plan are measurable and traceable to specific user requirements or code quality standards. No issues found with success criteria definitions.

### Planning Log Assessment

The planning log is thorough: 6 unaddressed research items (DR-01 through DR-06) with impact ratings, 3 documented deviations (DD-01 through DD-03) with rationale, 4 implementation paths considered with rejection reasoning, and 6 follow-on work items (WI-01 through WI-06) with sources and dependencies. All entries are well-structured and actionable.

### Clarifying Questions

1. **S2 tier scope**: Should the plan include an Azure AI Search tier recommendation in the README/documentation only, or should it also add tier-aware configuration (e.g., partition count, replica count) to `config.yaml`?
2. **requirements.md Section 12 update**: Is the user willing to amend Section 12 of `requirements.md` to explicitly permit Azure Cognitive Services (Florence) alongside Azure AI Foundry models? If not, should the plan switch to Cohere Embed v4 as primary?
3. **Sync wrapper pattern**: WI-04 flags `asyncio.run()` in sync wrappers as a Critical finding from an earlier review. Should this be fixed as part of this plan (Step 4.1) or remain deferred to WI-04?
