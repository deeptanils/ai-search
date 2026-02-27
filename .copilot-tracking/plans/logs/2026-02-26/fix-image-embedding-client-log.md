<!-- markdownlint-disable-file -->
# Planning Log: Fix Image Embedding Client

## Discrepancy Log

Gaps and differences identified between research findings and the implementation plan.

### Unaddressed Research Items

* DR-01: Endpoint routing uncertainty — whether `ImageEmbeddingsClient` uses the same base URL or requires a different endpoint
  * Source: embedding-vector-correctness-research.md (Lines 195-200, "Potential Next Research")
  * Reason: The SDK handles route differentiation internally (`/embeddings` vs `/images/embeddings`). The same base Foundry endpoint should work for both clients. This will be validated during Phase 3 (re-ingestion).
  * Impact: low — if routing fails, the error will surface immediately during re-ingestion and can be resolved by adjusting the endpoint

* DR-02: `input_type` optimization for text embeddings (query vs document)
  * Source: embedding-vector-correctness-research.md (Lines 170-180, "Text Embedding input_type")
  * Reason: This is an enhancement, not a bug fix. Included as optional Step 1.3 but not required for correctness.
  * Impact: low — text embeddings already work correctly; `input_type` may improve retrieval quality marginally

### Plan Deviations from Research

* DD-01: Relevance scoring thresholds not re-tuned in this plan
  * Research recommends: After fixing image embeddings, score distributions will change dramatically (from 0.95+ spread to 0.3-0.9 spread), which may require different thresholds in `relevance.py`
  * Plan implements: Validates new score distribution in Phase 3 but defers threshold re-tuning to follow-on work
  * Rationale: Threshold values depend on empirical data from the corrected embeddings. Re-tuning before seeing actual score distributions is premature. Current thresholds (z-score, gap ratio, spread) are relative metrics and may still work with wider score distributions.

* DD-02: Ingestion script `--force` flag vs index delete-recreate
  * Research recommends: Re-index all 10 documents after code fix
  * Plan implements: Two options — either add `--force` flag to `ingest_samples.py` or delete/recreate index. Implementer can choose.
  * Rationale: Both achieve the same result. Delete-recreate is simpler but loses any manual changes to the index schema. The `--force` flag is more surgical.

## Implementation Paths Considered

### Selected: Add `ImageEmbeddingsClient` factory alongside existing text client

* Approach: Add a new `get_foundry_image_embed_client()` factory in `clients.py` that returns `ImageEmbeddingsClient`. Keep `get_foundry_embed_client()` unchanged for text embeddings. Update `_embed_image_foundry()` to use the image client with `ImageEmbeddingInput`.
* Rationale: Minimal changes, follows existing code patterns, separates concerns clearly between text and image embedding
* Evidence: Research doc (Lines 125-165) confirms this is the correct SDK usage

### IP-01: Replace `EmbeddingsClient` with a unified multimodal client

* Approach: Investigate if there is a single client that handles both text and image embedding, eliminating the need for two factories
* Trade-offs: Would reduce client proliferation but the SDK explicitly separates these concerns into two classes (`EmbeddingsClient` and `ImageEmbeddingsClient`) with different input types and routes
* Rejection rationale: The SDK does not provide a unified client. The two clients serve fundamentally different routes (`/embeddings` vs `/images/embeddings`) and accept different input types (`List[str]` vs `List[ImageEmbeddingInput]`). Forcing unification would require an abstraction layer with no benefit.

### IP-02: Write a wrapper function that auto-detects content type

* Approach: Create a single `embed()` function that inspects the input (text vs data URI) and dispatches to the appropriate client automatically
* Trade-offs: Cleaner call site but adds implicit behavior and makes testing harder; the caller already knows whether they have text or image data
* Rejection rationale: The existing code structure already separates `embed_image()` from `embed_text_for_image_search()`. Adding auto-detection would introduce ambiguity (base64 strings could be mistaken for text) and make the code harder to reason about.

## Suggested Follow-On Work

Items identified during planning that fall outside current scope.

* WI-01: Re-tune relevance scoring thresholds — After re-ingestion with corrected image embeddings, collect score distributions for exact matches, similar content, and irrelevant queries. Use empirical data to set optimal thresholds in `relevance.py`. (medium priority)
  * Source: DD-01 deviation log
  * Dependency: Phase 3 completion (need actual score data)

* WI-02: Add `input_type` to text embedding calls — Set `input_type="query"` for search-time text embeddings and `input_type="document"` for ingestion-time text. (low priority)
  * Source: DR-02 research item
  * Dependency: None (can be done independently)

* WI-03: Test with larger corpus — Current validation uses only 10 documents. Test with 50-100+ images to validate that relevance scoring thresholds scale correctly. (medium priority)
  * Source: Relevance module note about small corpus limitations
  * Dependency: WI-01 completion

* WI-04: Add integration test for image embedding round-trip — Create a test that calls the actual Azure AI endpoint with a test image and validates the response dimensions and score differentiation. (low priority)
  * Source: General testing best practice
  * Dependency: None
