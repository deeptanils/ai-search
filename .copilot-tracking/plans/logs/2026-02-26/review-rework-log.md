<!-- markdownlint-disable-file -->
# Planning Log: Review Rework — Major Findings + Observability

## Discrepancy Log

Gaps and differences identified between research findings and the implementation plan.

### Unaddressed Research Items

* DR-01: IV-012 (Minor) — Empty text input validation for `embed_text_for_image_search()`
  * Source: Research doc, "Potential Next Research" section
  * Reason: Minor finding; outside current rework scope (only 3 major findings targeted)
  * Impact: Low — empty text sent to Florence returns a valid but meaningless vector

* DR-02: Sync wrapper `generate_query_vectors_sync` not tested
  * Source: Research doc, Scenario 1 "Considered Alternatives"
  * Reason: RPI-004 only requires tests for the async function; sync wrapper is a trivial `asyncio.run()` call
  * Impact: Low — single-line wrapper with no branching logic

### Plan Deviations from Research

* DD-01: Observability additions not in original research
  * Research recommends: No observability changes; research focused only on the 3 major findings
  * Plan implements: `structlog` warning/error logs on all new validation and guard error paths
  * Rationale: User explicitly requested "also consider observability and tracking"; structlog is already used across the codebase

* DD-02: Misconfiguration test placed in `test_image.py` instead of `test_config.py`
  * Research recommends: Either `test_image.py` or `test_config.py` (both listed as options)
  * Plan implements: `test_image.py` with a `TestSecretsValidation` class
  * Rationale: The test validates `get_cv_client()` behavior which is called from the image embedding flow; co-locating with related tests improves test discoverability

* DD-03: 6 new tests instead of research-suggested "5 new + 1 misconfiguration"
  * Research recommends: "5 new tests in test_image.py (3 for embed_image, 2 for embed_text_for_image_search)" plus "1 new test for misconfiguration error"
  * Plan implements: 3 embed_image validation + 2 embed_text_for_image_search validation + 1 secrets misconfiguration = 6 tests total (matches research count)
  * Rationale: No deviation — counts align; labeling clarified for tracking

* DD-04: Added `result: list[float] = vector` type annotation in `image.py`
  * Plan specifies: `return vector` directly after validation guard
  * Implementation differs: Added typed `result` variable to satisfy mypy `no-any-return` rule since `data.get()` returns `Any`
  * Rationale: mypy strict mode requires explicit type annotation; plan's code snippet did not account for this

## Implementation Paths Considered

### Selected: `data.get("vector")` + dimension check + `ValueError` + structlog (Option D+)

* Approach: Validate Florence response with `.get()`, check non-empty and 1024 dimensions, raise `ValueError` with diagnostic info, log `warning` before raising
* Rationale: Minimal code (~8 lines per function including logging), no new imports or exception types, follows existing `ValueError` + `msg` pattern, structlog already imported
* Evidence: Research doc, Scenario 2 — Option D selected; enriched with structlog per user request

### IP-01: Pydantic response model (Option B)

* Approach: Create `FlorenceResponse(BaseModel)` with `vector: list[float]` field, use `model_validate()` on response JSON
* Trade-offs: Type-safe validation with automatic deserialization, but adds a model class for a single-field response; overkill for current needs
* Rejection rationale: No other REST responses in codebase use Pydantic response models; Florence response is a single field; proportional benefit is low

### IP-02: Custom `FlorenceAPIError` exception (Option C)

* Approach: Define `FlorenceAPIError(Exception)` and raise instead of `ValueError`
* Trade-offs: More specific exception type for callers to catch, but no precedent for custom exceptions in codebase; all callers currently catch `ValueError`
* Rejection rationale: Codebase consistently uses `ValueError` with descriptive messages; introducing a custom exception adds API surface without matching convention

### IP-03: Secrets with `enabled: bool` flag (Option B for IV-014)

* Approach: Add `enabled: bool = False` field to `AzureComputerVisionSecrets`, check flag before loading
* Trade-offs: Explicit opt-in semantics, but adds a configuration concept not used by any other secrets class
* Rejection rationale: Over-engineered for the use case; `None` defaults with lazy validation achieve the same effect with less configuration surface

## Suggested Follow-On Work

Items identified during planning that fall outside current scope.

* WI-01: Add empty-text input validation to `embed_text_for_image_search()` — Straightforward `ValueError` check for empty/whitespace text input (Low priority)
  * Source: DR-01 / IV-012 minor finding
  * Dependency: None — can be done independently

* WI-02: Add tests for `generate_query_vectors_sync` wrapper — Test that `asyncio.run()` wrapper correctly delegates to async function (Low priority)
  * Source: DR-02 / Research Scenario 1
  * Dependency: RPI-004 test file must exist (Phase 3 of this plan)

* WI-03: Add `@lru_cache` to `lru_cache` clear in conftest or fixtures — Clear all cached clients between tests to prevent test pollution (Medium priority)
  * Source: Planning observation — `get_cv_client.cache_clear()` only done in one test
  * Dependency: None

* WI-04: Add structured metrics/tracing for Florence API latency — Track Florence API response times with structlog event fields or OpenTelemetry spans (Medium priority)
  * Source: User observability request — current logging tracks success/failure but not latency
  * Dependency: None — can be implemented independently

* WI-05: Address remaining minor review findings (IV-005, IV-009, IV-012, IV-013, IV-018-020, IV-026) — 8 minor findings from original review (Low priority)
  * Source: Review log
  * Dependency: Current rework should complete first

## Plan Validation

### Validation Run: 2026-02-26

**Status**: PASS (0 Critical, 0 Major, 2 Minor)

#### Findings

* PV-01 (Minor): Plan line number references for `image.py` reference "lines 59-64" and "lines 89-94" — actual validation target is lines 61-64 (`response.raise_for_status()` through `return vector`) and lines 91-94. The broader range is acceptable as it includes preceding context.
  * Impact: None — implementer will locate the correct lines from the code snippets provided

* PV-02 (Minor): Phase 2 marked `parallelizable: false` with dependency note "Depends on Phase 1 (secrets type changes affect get_cv_client() callers)" — however, the actual code changes in Phase 2 (`image.py` response validation) do not depend on Phase 1 (`config.py` / `clients.py` secrets changes). Only the misconfiguration test in Step 2.3 depends on Phase 1.
  * Impact: Low — sequential ordering is the conservative safe choice; no implementation risk

#### Summary

* All 4 user requirements addressed (RPI-004, IV-011, IV-014, observability)
* Research approaches match selected implementation paths
* All line number references verified against source files
* Test coverage includes all new error paths (6 tests) plus 3 query tests
* Dependencies correctly ordered
* Success criteria are measurable and traceable
