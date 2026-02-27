---
title: Hybrid Retrieval and Re-Ranking Strategies for Multi-Vector AI Image Search
description: Deep research on hybrid search, multi-vector queries, scoring profiles, two-stage retrieval, query expansion, performance, and diversity for Azure AI Search
author: research-subagent
ms.date: 2026-02-26
ms.topic: reference
keywords:
  - azure ai search
  - hybrid retrieval
  - reciprocal rank fusion
  - multi-vector search
  - re-ranking
  - scoring profiles
  - query expansion
  - mmr
estimated_reading_time: 25
---

## Executive Summary

This document presents research findings on hybrid retrieval and re-ranking strategies
for a multi-vector AI image search pipeline built on Azure AI Search. The pipeline
indexes images with three embedding vectors (semantic, structural, style) plus keyword
metadata, and must support sub-300ms P95 latency at 10M+ document scale. Key findings
include Azure AI Search's native Reciprocal Rank Fusion (RRF) for merging BM25 and
vector results, the `weight` parameter on `VectorizedQuery` for tuning vector
contributions, scoring profile design for metadata boosting, and a two-stage
retrieval architecture with application-level re-ranking.

---

## 1. Hybrid Search in Azure AI Search

### 1.1 How BM25 Combines with Vector Searches

Azure AI Search supports **hybrid search** by executing BM25 keyword retrieval and
vector similarity searches in parallel, then merging results using Reciprocal Rank
Fusion (RRF). When a search request includes both a `search` text parameter and one
or more `vectorQueries`, Azure AI Search:

1. Runs BM25 against all `searchable` text fields.
2. Runs each vector query independently against its target vector field using HNSW
   approximate nearest neighbor search (cosine similarity in our case).
3. Merges all result sets using RRF.

Each independent retriever (BM25 + N vector queries) produces a ranked list. RRF
combines these lists into a single ranked output. For our pipeline with three vector
fields plus BM25, there are **four parallel retrievers** feeding into RRF.

### 1.2 Reciprocal Rank Fusion (RRF)

RRF is a rank-based fusion method that does not depend on raw scores from individual
retrievers. The formula is:

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

Where:

* $d$ is a document
* $R$ is the set of all rankers (BM25, semantic vector, structural vector, style
  vector)
* $\text{rank}_r(d)$ is the rank of document $d$ in ranker $r$'s result list
* $k$ is a constant (Azure AI Search uses $k = 60$ by default)

**Key properties of RRF:**

* Rank-based, so it is robust to score distribution differences between retrievers.
* Documents appearing in multiple result lists get boosted.
* Documents ranked highly by any single retriever still surface.
* The constant $k = 60$ dampens the effect of very high ranks, making the fusion more
  balanced.

**Azure AI Search specifics:**

* RRF is the **only** fusion method available; you cannot switch to CombSUM, CombMNZ,
  or other fusion strategies natively.
* Each vector query is treated as a separate ranker input to RRF.
* BM25 is always treated as one ranker input when a `search` text parameter is provided.

### 1.3 The `weight` Parameter on VectorizedQuery

Starting with API version `2024-05-01-preview` and GA in `2024-07-01`, Azure AI Search
supports a `weight` parameter on each `VectorizedQuery`. This parameter controls the
**relative importance** of that vector query's contribution to RRF scoring.

**How it works:**

The weight acts as a multiplier on the reciprocal rank contribution. When computing
the RRF score for a document, each ranker's contribution is scaled by its weight:

$$\text{RRF}(d) = \sum_{r \in R} w_r \cdot \frac{1}{k + \text{rank}_r(d)}$$

Where $w_r$ is the weight assigned to ranker $r$.

**Default weight:** `1.0` for all vector queries.

**Configuration example:**

```python
from azure.search.documents.models import VectorizedQuery

vector_queries = [
    VectorizedQuery(
        vector=semantic_embedding,
        k_nearest_neighbors=50,
        fields="semantic_vector",
        weight=0.5,
    ),
    VectorizedQuery(
        vector=structural_embedding,
        k_nearest_neighbors=50,
        fields="structural_vector",
        weight=0.2,
    ),
    VectorizedQuery(
        vector=style_embedding,
        k_nearest_neighbors=50,
        fields="style_vector",
        weight=0.2,
    ),
]
```

**Important notes:**

* Weights are relative to each other, not absolute percentages.
* The weight range is 0.1 to 1000.
* Setting a weight of 0 effectively disables that vector query (but the minimum is
  0.1).

### 1.4 Controlling BM25 Weight Relative to Vector Weights

**This is a critical gap.** As of the current API versions, **there is no direct
`weight` parameter for the BM25 keyword ranker** in the RRF fusion. The `weight`
parameter exists only on `VectorizedQuery` objects.

**Workarounds to control BM25 influence:**

1. **Scoring profiles**: Apply a scoring profile that boosts or dampens BM25 scores
   based on field weights and functions. This indirectly affects BM25's rank ordering,
   which flows into RRF. However, it does not directly set a weight multiplier on the
   BM25 ranker in the RRF formula.

2. **Adjusting vector weights**: Since vector weights are relative, increasing all
   vector weights (for example, setting semantic to 5.0, structural to 2.0, style to
   2.0) proportionally diminishes BM25's influence in the RRF fusion because BM25's
   implicit weight remains at 1.0.

3. **Omitting the `search` parameter**: If you want to disable BM25 entirely, simply
   do not pass a `search` text query. This makes the search purely vector-based.

4. **`maxTextRecallSize` parameter**: Controls how many BM25 results feed into RRF
   (default 1000, range 1-10000). Reducing this effectively limits BM25's influence
   by reducing the candidate pool it contributes.

**Recommended approach for our pipeline:**

Given the config weights (`semantic_weight: 0.5`, `structural_weight: 0.2`,
`style_weight: 0.2`, `keyword_weight: 0.1`), set vector weights to 5.0, 2.0, 2.0
respectively. BM25 has an implicit weight of 1.0, so the effective ratio becomes
5:2:2:1, matching the desired 0.5:0.2:0.2:0.1 proportions.

---

## 2. Multi-Vector Query Strategy

### 2.1 Text Query Flow

When a user submits a text query:

1. **Semantic embedding**: Pass the raw query text to `text-embedding-3-large` via Azure
   AI Foundry. This produces a 3072-dimensional vector for the `semantic_vector` field.

2. **Structural embedding**: Use the LLM to generate a layout/composition description
   from the query (for example, "foreground subject left, background mountains, sky
   upper third"), then embed that description with the structural embedding model
   (1024-dimensional).

3. **Style embedding**: Use the LLM to extract style descriptors from the query (for
   example, "cinematic lighting, warm tones, film grain"), then embed with the style
   encoder (512-dimensional).

4. **BM25 keywords**: The raw query text (and potentially expanded terms) is passed as
   the `search` parameter for BM25 matching against `generation_prompt` and `tags`.

### 2.2 Image Query Flow

When a user submits an image:

1. Use **GPT-4o** (via Azure AI Foundry) to analyze the image and produce:
   * A detailed semantic description
   * A structural/layout description
   * A style description
   * Extracted keywords/tags

2. Generate embeddings from each description using the corresponding model.

3. Use the extracted keywords as the `search` parameter for BM25.

**Prompt template for image analysis:**

```text
Analyze this image and provide:
1. SEMANTIC: A detailed description of the scene, subjects, actions, and meaning.
2. STRUCTURAL: Describe the spatial layout, composition, and positioning of elements.
3. STYLE: Describe the artistic style, lighting, color palette, and mood.
4. KEYWORDS: A comma-separated list of relevant search keywords.

Respond in JSON format with keys: semantic, structural, style, keywords.
```

### 2.3 Issuing Parallel Vector Searches with Different Weights

Azure AI Search handles multiple vector queries in a single API call. All vector
queries within a single request are executed in parallel server-side and merged via RRF.

```python
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

results = search_client.search(
    search_text="cinematic dark alley rain",  # BM25
    vector_queries=[
        VectorizedQuery(
            vector=semantic_emb,
            k_nearest_neighbors=100,
            fields="semantic_vector",
            weight=5.0,
        ),
        VectorizedQuery(
            vector=structural_emb,
            k_nearest_neighbors=100,
            fields="structural_vector",
            weight=2.0,
        ),
        VectorizedQuery(
            vector=style_emb,
            k_nearest_neighbors=100,
            fields="style_vector",
            weight=2.0,
        ),
    ],
    top=200,
    select=["image_id", "generation_prompt", "scene_type",
            "emotional_polarity", "tags", "low_light_score",
            "narrative_intent", "required_objects"],
)
```

**Key design decisions:**

* Set `k_nearest_neighbors` to 100-200 per vector query. HNSW will retrieve this many
  candidates per vector field.
* Set `top` to your desired Stage 1 candidate count (100-500).
* Use `select` to retrieve only the fields needed for Stage 2 re-ranking, minimizing
  payload size.

### 2.4 Weight Mapping from config.yaml

| config.yaml Key      | Value | Vector Query Weight | Rationale               |
|-----------------------|-------|---------------------|-------------------------|
| `semantic_weight`     | 0.5   | 5.0                 | Primary retrieval signal |
| `structural_weight`   | 0.2   | 2.0                 | Layout matching          |
| `style_weight`        | 0.2   | 2.0                 | Artistic style matching  |
| `keyword_weight`      | 0.1   | 1.0 (implicit BM25) | Text keyword fallback    |

The multiplier of 10x is applied consistently so the ratio is preserved, and BM25's
default weight of 1.0 naturally maps to the 0.1 proportion.

---

## 3. Scoring Profile Design

### 3.1 Azure AI Search Scoring Profiles for BM25 Boosting

Scoring profiles modify the BM25 relevance score before it enters RRF. They support:

* **Field weights**: Boost specific searchable fields in BM25.
* **Scoring functions**: Apply mathematical functions based on field values.

**Available function types:**

| Function    | Description                                        | Use Case                            |
|-------------|----------------------------------------------------|-------------------------------------|
| `magnitude` | Boost based on numeric range                       | `emotional_polarity`, `low_light`   |
| `freshness` | Boost based on date recency                        | Recently indexed images             |
| `distance`  | Boost based on geographic proximity                | Not applicable here                 |
| `tag`       | Boost based on tag matching                        | `tags` field matching               |

### 3.2 Scoring Profile for Metadata Boosting

```json
{
  "name": "metadata-boost",
  "text": {
    "weights": {
      "generation_prompt": 2.0,
      "tags": 1.5
    }
  },
  "functions": [
    {
      "type": "magnitude",
      "fieldName": "emotional_polarity",
      "boost": 2,
      "interpolation": "linear",
      "magnitude": {
        "boostingRangeStart": 0,
        "boostingRangeEnd": 1,
        "constantBoostBeyondRange": true
      }
    },
    {
      "type": "tag",
      "fieldName": "tags",
      "boost": 1.5,
      "tag": {
        "tagsParameter": "query_tags"
      }
    }
  ],
  "functionAggregation": "sum"
}
```

**Usage in search request:**

```python
results = search_client.search(
    search_text=query_text,
    scoring_profile="metadata-boost",
    scoring_parameters=["query_tags-cinematic,dark,rain"],
    vector_queries=[...],
    top=200,
)
```

### 3.3 Can Scoring Profiles and Vector Weights Coexist?

**Yes.** Scoring profiles and vector weights operate on different components:

* **Scoring profiles** modify BM25 scores, which change the BM25 rank ordering before
  it enters RRF.
* **Vector weights** scale each vector query's RRF contribution.

They are complementary and do not conflict. The flow is:

1. BM25 computes base scores.
2. Scoring profile modifies BM25 scores (boosting certain fields/functions).
3. BM25 results are ranked by modified scores.
4. Each vector query produces its own ranked list.
5. RRF merges all ranked lists, applying vector weights and the implicit BM25 weight.

**Important caveat:** Scoring profiles do NOT affect vector search scores. They only
modify the text-based (BM25) portion of hybrid search.

**Recommendation:** Use scoring profiles conservatively. Heavy BM25 boosting can
distort rank ordering and hurt RRF fusion quality. Since BM25 carries only 10% weight
in our pipeline, scoring profiles have limited but useful impact for tie-breaking and
metadata-aware boosting within the BM25 ranker.

---

## 4. Two-Stage Retrieval and Re-Ranking

### 4.1 Architecture Overview

```text
┌─────────────────────────────────────────────────┐
│  Stage 1: Azure AI Search (Hybrid + RRF)        │
│  - BM25 + 3 vector queries                      │
│  - Returns top-K candidates (K=200)             │
│  - Latency budget: ~150ms                       │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  Stage 2: Application-Level Re-Ranking          │
│  - Emotional alignment scoring                  │
│  - Narrative consistency scoring                 │
│  - Object overlap (Jaccard)                     │
│  - Low-light compatibility                      │
│  - Final score = weighted combination           │
│  - Returns top-N results (N=10-50)              │
│  - Latency budget: ~100ms                       │
└─────────────────────────────────────────────────┘
```

### 4.2 Stage 1: Azure AI Search Retrieval

* **K parameter**: Retrieve 200 candidates as baseline. At 10M documents with HNSW,
  `k_nearest_neighbors=100` per vector field plus BM25 `maxTextRecallSize=1000` is
  sufficient to surface strong candidates.
* **Select only re-ranking fields**: Request only the metadata fields needed for
  Stage 2 (avoid transferring full embeddings back).
* **Use filters sparingly**: Pre-filters (`$filter`) reduce the candidate pool before
  vector search. Useful for hard constraints like `scene_type eq 'exterior'` but should
  not be overused as they can hurt recall.

### 4.3 Stage 2: Application-Level Re-Ranking

#### 4.3.1 Emotional Alignment

Compare the query's emotional trajectory with each candidate's stored emotional
metadata:

```python
def emotional_alignment_score(
    query_emotions: dict, candidate_emotions: dict
) -> float:
    """Compute emotional alignment between query and candidate.

    Both dicts have keys: starting_emotion, mid_emotion, end_emotion,
    emotional_polarity.
    """
    polarity_diff = abs(
        query_emotions["emotional_polarity"]
        - candidate_emotions["emotional_polarity"]
    )
    polarity_score = 1.0 - polarity_diff

    # Categorical match for emotion labels
    emotion_keys = ["starting_emotion", "mid_emotion", "end_emotion"]
    matches = sum(
        1 for k in emotion_keys
        if query_emotions.get(k, "").lower()
        == candidate_emotions.get(k, "").lower()
    )
    trajectory_score = matches / len(emotion_keys)

    return 0.6 * polarity_score + 0.4 * trajectory_score
```

#### 4.3.2 Narrative Consistency

Compare narrative intent labels or use embedding similarity:

```python
def narrative_consistency_score(
    query_narrative: str, candidate_narrative: str
) -> float:
    """Score narrative alignment.

    Option A: Exact/fuzzy label match.
    Option B: Embedding cosine similarity (pre-computed).
    """
    if query_narrative.lower() == candidate_narrative.lower():
        return 1.0

    # Fuzzy matching using token overlap
    query_tokens = set(query_narrative.lower().split())
    candidate_tokens = set(candidate_narrative.lower().split())
    if not query_tokens or not candidate_tokens:
        return 0.0
    overlap = len(query_tokens & candidate_tokens)
    return overlap / max(len(query_tokens), len(candidate_tokens))
```

#### 4.3.3 Object Overlap (Jaccard Similarity)

```python
def object_overlap_score(
    query_objects: list[str], candidate_objects: list[str]
) -> float:
    """Jaccard similarity between required object sets."""
    q_set = {o.lower() for o in query_objects}
    c_set = {o.lower() for o in candidate_objects}
    if not q_set and not c_set:
        return 1.0
    if not q_set or not c_set:
        return 0.0
    intersection = len(q_set & c_set)
    union = len(q_set | c_set)
    return intersection / union
```

#### 4.3.4 Low-Light Compatibility

```python
def low_light_compatibility_score(
    query_low_light: float | None, candidate_low_light: float
) -> float:
    """Score low-light compatibility.

    If query doesn't specify low-light preference, return neutral score.
    """
    if query_low_light is None:
        return 0.5  # Neutral
    diff = abs(query_low_light - candidate_low_light)
    return 1.0 - diff
```

#### 4.3.5 Combined Re-Ranking Score

```python
def compute_rerank_score(
    candidate: dict,
    query_context: dict,
    weights: dict | None = None,
) -> float:
    """Compute final re-ranking score for a candidate."""
    if weights is None:
        weights = {
            "emotional": 0.3,
            "narrative": 0.25,
            "object_overlap": 0.25,
            "low_light": 0.2,
        }

    scores = {
        "emotional": emotional_alignment_score(
            query_context["emotions"], candidate["emotions"]
        ),
        "narrative": narrative_consistency_score(
            query_context["narrative_intent"],
            candidate["narrative_intent"],
        ),
        "object_overlap": object_overlap_score(
            query_context["required_objects"],
            candidate["required_objects"],
        ),
        "low_light": low_light_compatibility_score(
            query_context.get("low_light_score"),
            candidate["low_light_score"],
        ),
    }

    return sum(weights[k] * scores[k] for k in weights)
```

#### 4.3.6 Efficient Python Implementation

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def rerank_candidates(
    candidates: list[dict],
    query_context: dict,
    top_n: int = 20,
) -> list[dict]:
    """Re-rank candidates efficiently using thread pool."""
    loop = asyncio.get_event_loop()

    with ThreadPoolExecutor(max_workers=4) as executor:
        scores = await loop.run_in_executor(
            executor,
            lambda: [
                (c, compute_rerank_score(c, query_context))
                for c in candidates
            ],
        )

    scored = sorted(scores, key=lambda x: x[1], reverse=True)
    return [
        {**c, "rerank_score": s} for c, s in scored[:top_n]
    ]
```

For 200 candidates with simple arithmetic scoring, the re-ranking is
compute-trivial and completes in under 5ms in Python. No thread pool is
strictly necessary at this scale, but it provides headroom for growth.

### 4.4 Should Stage 2 Use an LLM for Reasoning-Based Re-Rank?

**Analysis:**

| Approach             | Latency     | Cost        | Quality       | Complexity |
|----------------------|-------------|-------------|---------------|------------|
| Rule-based scoring   | < 5ms       | Zero        | Good          | Low        |
| Embedding similarity | < 10ms      | Minimal     | Good          | Low        |
| LLM re-ranking      | 500ms-3000ms | $0.01-0.10/query | Excellent | High       |

**Recommendation: Do NOT use LLM re-ranking in the real-time path.** The P95 latency
requirement of 300ms makes LLM re-ranking infeasible for the primary query flow. Even
with a small model, a single LLM call typically takes 500ms+.

**However, consider LLM re-ranking for:**

* **Offline evaluation**: Compare LLM rankings against rule-based rankings to tune
  weights.
* **Premium tier**: An optional "deep search" mode with relaxed latency SLAs.
* **Feedback loop**: Use LLM to periodically score samples and update rule-based
  weights.
* **Future extension**: The requirements mention "online re-ranking with LLM reasoning"
  as a future enhancement. Architect the system to support it as an optional async
  post-processing step.

---

## 5. Query Expansion

### 5.1 LLM-Based Query Expansion

Use the LLM to expand the user query before embedding, improving recall by covering
synonyms, related concepts, and implicit intent.

**Expansion prompt template:**

```text
Given the search query: "{query}"

Generate 3 expanded variants that capture different aspects of the intent:
1. A more detailed version with additional descriptive terms
2. A version focusing on visual/structural elements
3. A version focusing on mood, style, and artistic qualities

Also list 5-10 additional keywords that should be included in a keyword search.

Respond in JSON: {
  "detailed": "...",
  "structural": "...",
  "stylistic": "...",
  "additional_keywords": ["...", "..."]
}
```

**Integration strategy:**

1. Generate expanded queries via LLM.
2. Embed the "detailed" variant for `semantic_vector` search.
3. Embed the "structural" variant for `structural_vector` search.
4. Embed the "stylistic" variant for `style_vector` search.
5. Combine original query + `additional_keywords` for BM25 search text.

### 5.2 Multiple Query Variants for Better Recall

**Approach: Multi-query retrieval.** Generate multiple embedding variants and either:

* **Option A**: Issue separate vector queries per variant (Azure AI Search handles
  RRF merge). This is straightforward but increases the number of vector queries per
  request.
* **Option B**: Average/pool multiple embeddings into a single query vector. This
  reduces API calls but may lose specificity.

**Recommendation:** Use Option A with Azure AI Search's native multi-query support.
You can pass up to 10 vector queries in a single request. For three embedding types
with two variants each, that is six vector queries, well within limits.

### 5.3 Latency Impact of Query Expansion

LLM query expansion adds 300-800ms for the LLM call. Strategies to mitigate:

* **Cache expanded queries**: Hash the query text and cache LLM expansion results.
  Identical or near-identical queries get instant expansion.
* **Parallel execution**: Start embedding generation for the original query immediately.
  Run LLM expansion in parallel. If expansion completes in time, use expanded
  embeddings; otherwise, fall back to original.
* **Pre-compute for common queries**: Maintain a lookup table of pre-expanded common
  queries.

---

## 6. Performance Considerations

### 6.1 P95 Latency Budget (< 300ms)

**Latency breakdown estimate:**

| Component                    | Estimated P95 | Notes                                |
|------------------------------|---------------|--------------------------------------|
| Embedding generation         | 50-80ms       | `text-embedding-3-large` via Foundry |
| Azure AI Search hybrid query | 80-150ms      | HNSW + BM25 + RRF                   |
| Network overhead             | 20-40ms       | Between services                     |
| Stage 2 re-ranking           | 2-5ms         | Rule-based, in-process              |
| Response serialization       | 5-10ms        | JSON response to client              |
| **Total**                    | **157-285ms** | Within 300ms budget                  |

**Warning:** Query expansion adds 300-800ms and would blow the budget. It must either
be cached or run as an async pre-processing step.

### 6.2 Parallel vs Sequential Vector Searches

**Azure AI Search handles this natively.** All vector queries within a single search
request are executed in parallel on the search service side. There is no benefit to
issuing separate API calls; in fact, a single request with multiple vector queries is
always faster than multiple sequential requests because:

* Single network round-trip vs multiple
* Server-side parallel execution with shared result merging
* RRF fusion happens server-side without additional API calls

**Recommendation:** Always use a single search request with multiple `vectorQueries`.

### 6.3 Embedding Generation Parallelism

The three embeddings (semantic, structural, style) can be generated in parallel on the
client side using `asyncio.gather`:

```python
import asyncio

async def generate_all_embeddings(
    query_text: str, structural_desc: str, style_desc: str
) -> tuple[list[float], list[float], list[float]]:
    """Generate all three embeddings in parallel."""
    semantic_task = generate_embedding(query_text, model="text-embedding-3-large")
    structural_task = generate_embedding(structural_desc, model="structural-model")
    style_task = generate_embedding(style_desc, model="style-model")

    return await asyncio.gather(semantic_task, structural_task, style_task)
```

### 6.4 Caching Strategies

| Cache Layer              | What to Cache                       | TTL       | Backend         |
|--------------------------|-------------------------------------|-----------|-----------------|
| Query embedding cache    | Query text hash to embedding vector | 1 hour    | Redis           |
| LLM expansion cache      | Query text hash to expanded queries | 24 hours  | Redis           |
| Search result cache      | Full query hash to search results   | 5 minutes | Redis / in-mem  |
| Scoring profile cache    | Profile name to config              | On deploy | In-memory       |

**Cache key design:**

```python
import hashlib
import json

def cache_key(query_text: str, weights: dict) -> str:
    """Generate deterministic cache key."""
    payload = json.dumps({"q": query_text, "w": weights}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()
```

**Important:** Cache invalidation must account for index updates. When new images are
indexed, search result caches should be invalidated or given short TTLs.

### 6.5 HNSW Configuration for Performance

| Parameter        | Recommended Value | Impact                                       |
|------------------|-------------------|----------------------------------------------|
| `m`              | 4                 | Connections per node. Lower = faster, less accurate |
| `efConstruction` | 400               | Build quality. Higher = better recall         |
| `efSearch`       | 500               | Search quality. Higher = better recall, slower|
| `metric`         | cosine            | Required for normalized text embeddings       |

For 10M documents, HNSW with these settings provides ~95%+ recall@100 with sub-100ms
vector search latency per query.

---

## 7. Candidate Diversity

### 7.1 MMR (Maximal Marginal Relevance)

MMR balances relevance and diversity by iteratively selecting documents that are both
relevant to the query and dissimilar to already-selected documents:

$$\text{MMR}(d) = \lambda \cdot \text{Sim}(d, q) - (1 - \lambda) \cdot \max_{d_j \in S} \text{Sim}(d, d_j)$$

Where:

* $\lambda$ controls the relevance-diversity trade-off (0.5-0.7 typical)
* $\text{Sim}(d, q)$ is relevance to the query
* $\text{Sim}(d, d_j)$ is similarity to already-selected documents
* $S$ is the set of already-selected results

### 7.2 MMR Implementation

```python
import numpy as np
from numpy.typing import NDArray

def mmr_rerank(
    query_embedding: NDArray[np.float32],
    candidate_embeddings: NDArray[np.float32],
    candidate_scores: list[float],
    lambda_param: float = 0.6,
    top_n: int = 20,
) -> list[int]:
    """Apply MMR to diversify results.

    Args:
        query_embedding: Query vector (D,).
        candidate_embeddings: Candidate matrix (N, D).
        candidate_scores: Relevance scores from Stage 2.
        lambda_param: Relevance vs diversity trade-off.
        top_n: Number of results to return.

    Returns:
        List of indices into candidate arrays, ordered by MMR.
    """
    n = len(candidate_scores)
    selected: list[int] = []
    remaining = set(range(n))

    # Normalize scores to [0, 1]
    max_score = max(candidate_scores) if candidate_scores else 1.0
    norm_scores = [s / max_score for s in candidate_scores]

    # Pre-compute similarities between all candidates
    # Using cosine similarity (assumes normalized embeddings)
    sim_matrix = candidate_embeddings @ candidate_embeddings.T

    for _ in range(min(top_n, n)):
        best_idx = -1
        best_mmr = -float("inf")

        for idx in remaining:
            relevance = norm_scores[idx]

            if selected:
                max_sim = max(sim_matrix[idx, j] for j in selected)
            else:
                max_sim = 0.0

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        if best_idx >= 0:
            selected.append(best_idx)
            remaining.discard(best_idx)

    return selected
```

### 7.3 When to Apply MMR

MMR should be applied as a **Stage 3** step after Stage 2 re-ranking:

1. **Stage 1**: Azure AI Search returns 200 candidates.
2. **Stage 2**: Re-rank by emotional/narrative/object/low-light scoring, get top 50.
3. **Stage 3**: Apply MMR to top 50, return top 20 diverse results.

**Performance note:** MMR on 50 candidates with pre-computed similarity matrix takes
< 1ms. The bottleneck is fetching the embeddings if they are not already available.

**Design consideration:** To avoid fetching full embeddings for MMR, use semantic
vectors only (the primary representation). Include `semantic_vector` in the `select`
fields of the Stage 1 query, or store a compressed version.

### 7.4 Alternative Diversity Methods

| Method                | Pros                           | Cons                         |
|-----------------------|--------------------------------|------------------------------|
| MMR                   | Well-understood, tunable       | O(N*K) complexity            |
| Determinantal Point Processes (DPP) | Theoretically optimal | Complex, slower      |
| Clustering-based      | Fast, grouping-aware           | Requires cluster precompute  |
| Score threshold pruning | Simple                       | Does not explicitly diversify|

**Recommendation:** MMR is the best balance of quality, speed, and implementation
simplicity for our use case.

---

## 8. End-to-End Query Flow Summary

```text
User Query (text or image)
    │
    ├── [If image] GPT-4o analysis → descriptions + keywords
    │
    ├── [Optional] LLM query expansion (cached)
    │
    ├── Parallel embedding generation (3 vectors)
    │
    ▼
Stage 1: Azure AI Search
    ├── BM25 keyword search (weight: implicit 1.0)
    ├── Semantic vector search (weight: 5.0)
    ├── Structural vector search (weight: 2.0)
    └── Style vector search (weight: 2.0)
    │
    └── RRF fusion → top 200 candidates
    │
    ▼
Stage 2: Application Re-Ranking
    ├── Emotional alignment (0.3)
    ├── Narrative consistency (0.25)
    ├── Object overlap / Jaccard (0.25)
    └── Low-light compatibility (0.2)
    │
    └── Re-ranked → top 50
    │
    ▼
Stage 3: Diversity (MMR)
    └── λ = 0.6 → top 20 diverse results
    │
    ▼
Response to Client
```

---

## 9. Key Findings Summary

1. **RRF is rank-based and robust**: Azure AI Search's RRF fusion naturally handles
   different score distributions across BM25 and vector retrievers. No score
   normalization is needed.

2. **Vector weights map to config proportions**: Setting vector weights to 5.0, 2.0,
   2.0 with BM25's implicit 1.0 achieves the desired 0.5:0.2:0.2:0.1 ratio.

3. **No direct BM25 weight control**: The BM25 ranker does not have an explicit weight
   parameter in RRF. Workaround: scale vector weights proportionally.

4. **Scoring profiles complement vector weights**: They modify BM25 ranking (not
   vector) and are useful for metadata-aware boosting within the text retrieval
   component.

5. **Two-stage retrieval is essential**: Azure AI Search handles coarse retrieval;
   application-level re-ranking handles domain-specific scoring (emotional, narrative,
   object, low-light).

6. **LLM re-ranking is too slow for real-time**: At 500ms+ per call, it exceeds the
   300ms budget. Reserve it for offline evaluation and future premium tiers.

7. **Query expansion improves recall but adds latency**: Must be cached or run
   asynchronously. Pre-expansion for common queries is recommended.

8. **MMR provides diversity cheaply**: Applied as a post-re-ranking step on 50
   candidates, it adds negligible latency (< 1ms).

9. **Single-request multi-vector is optimal**: Azure AI Search parallelizes vector
   queries server-side within a single API call. Never issue separate requests.

10. **Latency budget is tight but feasible**: With parallel embedding generation and
    cached query expansion, the 300ms P95 target is achievable.

---

## 10. Recommended Next Research Topics

1. **Embedding model selection and benchmarking**: Compare `text-embedding-3-large` vs
   `text-embedding-3-small` vs multimodal CLIP variants for each vector field. Evaluate
   recall@K on a test set.

2. **Azure AI Search index sizing and cost modeling**: Estimate storage, compute, and
   cost for 10M documents with three vector fields of dimensions 3072, 1024, and 512.

3. **Semantic ranker (L2 re-ranker)**: Azure AI Search offers a built-in semantic
   ranker that applies a cross-encoder model server-side. Research whether it can
   replace or supplement Stage 2 re-ranking for the text retrieval component.

4. **Filter optimization**: Research the impact of pre-filters (for example,
   `scene_type`, `lighting_condition`) on recall and latency. Understand how HNSW
   behaves with filtered queries (pre-filter vs post-filter).

5. **Evaluation framework**: Design metrics (NDCG@K, MRR, precision@K, diversity@K)
   and a ground-truth labeling strategy for measuring retrieval quality.

6. **Character sub-vector indexing**: Research how to index and query nested
   character-level vectors (semantic, emotion, pose) within Azure AI Search. This may
   require flattening or a separate index.

7. **Compression and quantization**: Investigate scalar quantization and binary
   quantization options in Azure AI Search to reduce storage and improve latency for
   high-dimensional vectors.

8. **A/B testing framework**: Design an experimentation system to compare different
   weight configurations, scoring profiles, and re-ranking strategies in production.
