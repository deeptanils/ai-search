<!-- markdownlint-disable-file -->
# Task Research: Candidate Generation & AI Search Pipeline

Research into the architecture, technology choices, SDK capabilities, and implementation strategy for building a production-grade Candidate Generation and AI-powered Search pipeline using Azure AI Foundry and Azure AI Search.

## Task Implementation Requests

* Design and implement an end-to-end pipeline for candidate generation and AI-powered search
* Accept image + generation prompt as input, generate synthetic metadata via LLM
* Extract structured narrative, emotional, and structural information from images
* Generate multi-vector embeddings via Azure AI Foundry
* Index into Azure AI Search with hybrid retrieval and candidate re-ranking
* Python project with UV dependency management

## Scope and Success Criteria

* Scope: Full pipeline from ingestion through retrieval, using Azure AI Foundry for models and Azure AI Search for indexing/retrieval. Python codebase with UV.
* Assumptions:
  * Azure AI Foundry endpoint and API keys will be provided at deployment
  * Azure AI Search service is provisioned
  * Models available: gpt-4o (LLM), text-embedding-3-large (embeddings)
  * Future multimodal embeddings (CLIP/SigLIP) may be added later
* Success Criteria:
  * Clear architecture for each pipeline stage
  * Validated Azure AI Foundry SDK usage for LLM + embeddings
  * Validated Azure AI Search SDK for multi-vector index with HNSW + cosine
  * Hybrid search strategy with configurable weights
  * Project structure aligns with requirements (src/, config.yaml, .env, pyproject.toml, tests/)
  * Scalability considerations for 10M+ images

## Outline

1. Azure AI Foundry SDK research (LLM calls, embedding generation, multimodal)
2. Azure AI Search SDK research (index creation, multi-vector fields, hybrid search, scoring profiles)
3. Multi-vector encoding strategy (semantic, structural, style, character sub-vectors)
4. Hybrid retrieval and re-ranking approach
5. Python project setup with UV
6. Repository structure and configuration management
7. Alternatives analysis and selected approach

---

## Potential Next Research

* **PTU cost modeling**: Breakeven analysis for Provisioned Throughput Units vs pay-per-token at 10M image scale
* **Scalar/binary quantization**: Azure AI Search SDK 11.6.0+ supports compressed vector storage for ~50% memory reduction
* **Index alias management**: Zero-downtime schema migrations via index aliases
* **Evaluation framework**: NDCG@K, MRR, precision@K, diversity@K design and ground-truth labeling strategy
* **GPT-4o-mini cost optimization**: Using GPT-4o-mini for simpler images in extraction, GPT-4o for complex scenes
* **Docker containerization with UV**: Production image builds using UV
* **CI/CD pipeline**: GitHub Actions with UV for testing, linting, type checking
* **Async pipeline design**: `asyncio` + `httpx` patterns for parallel processing
* **A/B testing framework**: Experimentation for weight and re-ranking tuning in production

---

## Research Executed

### Subagent Research Documents

| Topic | Document | Status |
|-------|----------|--------|
| Azure AI Foundry SDK | [azure-ai-foundry-sdk-research.md](../subagents/2026-02-26/azure-ai-foundry-sdk-research.md) | Complete |
| Azure AI Search SDK | [azure-ai-search-sdk-research.md](../subagents/2026-02-26/azure-ai-search-sdk-research.md) | Complete |
| UV Python Project Setup | [uv-python-project-research.md](../subagents/2026-02-26/uv-python-project-research.md) | Complete |
| Multi-Vector Encoding | [multi-vector-encoding-research.md](../subagents/2026-02-26/multi-vector-encoding-research.md) | Complete |
| Hybrid Retrieval Strategy | [hybrid-retrieval-research.md](../subagents/2026-02-26/hybrid-retrieval-research.md) | Complete |

### Workspace Analysis

* **Current state**: Empty workspace — only `requirements.md` exists. No `pyproject.toml`, `src/`, `tests/`, `config.yaml`, or `.env`.
* **Requirements versions**: v1 (general architecture) and v2 (Azure-specific constraints, UV, config.yaml/.env split) both present in `requirements.md`.

---

## Key Discoveries

### 1. SDK and Package Selections

| Capability | Package | Version | Rationale |
|------------|---------|---------|-----------|
| LLM + Embeddings | `openai` | `>=1.58.0` | `AzureOpenAI` client — most mature, best documented for Azure AI Foundry |
| Azure Search | `azure-search-documents` | `>=11.6.0` | Full multi-vector + HNSW + hybrid search + scalar quantization support |
| Auth (production) | `azure-identity` | `>=1.17.0` | Entra ID token-based auth for production deployments |
| Data validation | `pydantic` | `>=2.0` | Structured Output schemas + data models |
| Settings | `pydantic-settings` | `>=2.0` | Auto-reads `.env` and env vars with prefix matching |
| Config | `pyyaml` | `>=6.0` | Non-secret YAML config loading |

* **`azure-ai-inference` is NOT needed** unless the project adopts non-OpenAI models from the Azure AI Model Catalog in the future.
* **API version `2024-12-01-preview`** supports all required features: vision, structured outputs, embeddings with dimensions, and Batch API.

### 2. GPT-4o as Unified Extraction Engine

A **single structured GPT-4o vision call** per image extracts all needed descriptions:

* Rich semantic description (for `semantic_vector`)
* Spatial/compositional analysis (for `structural_vector`)
* Artistic style description (for `style_vector`)
* Per-character: semantic, emotion, pose descriptions (for character sub-vectors)
* Synthetic metadata (scene_type, tags, lighting_condition, etc.)

This replaces the need for separate DINOv2 (structural) and Style Encoder (style) models. GPT-4o captures ~80-85% of structural quality and ~75-80% of style quality compared to dedicated vision encoders — sufficient for initial implementation with the Azure AI Foundry constraint.

**Structured Outputs**: Use `client.beta.chat.completions.parse()` with Pydantic models for schema-enforced JSON output. Requires `openai>=1.50.0`.

### 3. Matryoshka Embeddings for Multi-Dimensional Vectors

`text-embedding-3-large` supports the `dimensions` parameter via Matryoshka Representation Learning — a single model produces vectors at different dimensionalities:

| Vector Field | Dimensions | Quality Loss | Storage/Vector |
|-------------|------------|-------------|----------------|
| `semantic_vector` | 3072 | 0% (baseline) | 12 KB |
| `structural_vector` | 1024 | ~1% | 4 KB |
| `style_vector` | 512 | ~3% | 2 KB |
| `char_N_semantic_vector` | 512 | ~3% | 2 KB |
| `char_N_emotion_vector` | 256 | ~7% | 1 KB |
| `char_N_pose_vector` | 256 | ~7% | 1 KB |

**Per-image total (3 character slots)**: 7,680 dimensions → ~30 KB per image → ~280 GB for 10M images (vectors only).

**Embedding batching**: Group texts by target dimension (4 API calls per image, parallelizable). Total per image: 1 GPT-4o call + 4 embedding calls = 5 API calls.

### 4. Character Vectors Must Be Flattened

**Critical finding**: Azure AI Search does NOT support vector search on fields inside `Collection(Edm.ComplexType)`. Character vectors must be stored as top-level fields:

* `char_0_semantic_vector` (512 dims), `char_0_emotion_vector` (256 dims), `char_0_pose_vector` (256 dims)
* `char_1_semantic_vector`, `char_1_emotion_vector`, `char_1_pose_vector`
* `char_2_semantic_vector`, `char_2_emotion_vector`, `char_2_pose_vector`
* `character_count` (filterable integer)
* `character_metadata` (complex type for IDs and descriptors)

**Capped at 3 character slots** — overflow stored in metadata JSON for LLM re-ranking.

### 5. Hybrid Search Architecture (RRF + Weight Mapping)

Azure AI Search merges BM25 + vector results via **Reciprocal Rank Fusion (RRF)** with k=60. Vector query weights control RRF contributions.

**Config weight mapping**:

| config.yaml Key | Value | Vector Query `weight` | Explanation |
|----------------|-------|----------------------|-------------|
| `semantic_weight` | 0.5 | 5.0 | Primary retrieval signal |
| `structural_weight` | 0.2 | 2.0 | Layout matching |
| `style_weight` | 0.2 | 2.0 | Artistic style matching |
| `keyword_weight` | 0.1 | 1.0 (implicit BM25) | Text keyword fallback |

Multiply config weights by 10x to set vector weights, preserving ratio against BM25's implicit 1.0 weight. **No direct BM25 weight parameter exists** — this is the recommended workaround.

### 6. Three-Stage Retrieval Pipeline

```
Stage 1: Azure AI Search (Hybrid + RRF)
├── BM25 + 3 vector queries → top 200 candidates
├── Latency: ~150ms
│
Stage 2: Application-Level Re-Ranking
├── Emotional alignment (0.3), Narrative consistency (0.25)
├── Object overlap/Jaccard (0.25), Low-light compatibility (0.2)
├── → top 50 candidates
├── Latency: <5ms
│
Stage 3: MMR Diversity
├── λ=0.6, applied on top 50 → top 20 diverse results
├── Latency: <1ms
│
Total estimated P95: ~225ms (within 300ms budget)
```

**LLM re-ranking is NOT viable in real-time path** (500ms+ per call). Reserved for offline evaluation and future premium tier.

### 7. Project Structure with UV

```
ai-search/
├── src/
│   └── ai_search/              # Importable package (underscore)
│       ├── __init__.py
│       ├── config.py            # Pydantic settings + YAML loader
│       ├── models.py            # Shared Pydantic models
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── cli.py
│       │   ├── loader.py
│       │   └── metadata.py      # LLM metadata generation
│       ├── extraction/
│       │   ├── __init__.py
│       │   ├── narrative.py
│       │   ├── emotion.py
│       │   ├── objects.py
│       │   └── low_light.py
│       ├── embeddings/
│       │   ├── __init__.py
│       │   ├── semantic.py
│       │   ├── structural.py
│       │   ├── style.py
│       │   └── character.py
│       ├── indexing/
│       │   ├── __init__.py
│       │   ├── cli.py
│       │   ├── schema.py        # Azure Search index definition
│       │   └── indexer.py       # Document upload
│       └── retrieval/
│           ├── __init__.py
│           ├── cli.py
│           ├── query.py         # Query generation
│           ├── search.py        # Azure Search client
│           └── reranker.py      # Re-ranking + MMR
├── tests/
│   ├── conftest.py
│   ├── test_ingestion/
│   ├── test_extraction/
│   ├── test_embeddings/
│   ├── test_indexing/
│   └── test_retrieval/
├── config.yaml
├── .env.example
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

* **Build backend**: `hatchling` (UV default), `packages = ["src/ai_search"]`
* **Dev dependencies**: PEP 735 `[dependency-groups]` for pytest, ruff, mypy
* **Imports**: `from ai_search.config import Settings` (no sys.path hacks)
* **`src/` has NO `__init__.py`** — only `src/ai_search/` and subpackages do

### 8. Configuration Management Pattern

* **`.env`** (secrets via `pydantic-settings`): `AZURE_FOUNDRY_ENDPOINT`, `AZURE_FOUNDRY_API_KEY`, `AZURE_AI_SEARCH_ENDPOINT`, `AZURE_AI_SEARCH_API_KEY`
* **`config.yaml`** (non-secrets via PyYAML): model names, search weights, vector dimensions, HNSW parameters
* **`pydantic-settings`** auto-reads `.env` and env vars with prefix matching
* **`.env.example`** committed to git with placeholder values

### 9. Batch Processing for 10M+ Images

* **Azure OpenAI Batch API**: 50% cost reduction, 24h SLA, handles millions of requests asynchronously
* **Batch embedding**: Up to 2048 texts per API call
* **Index upload**: 500 docs/batch (with multi-vector fields), parallel workers with exponential backoff
* **PTU recommended**: Provisioned Throughput Units for guaranteed capacity at scale
* **Estimated cost**: $0.01-0.03 per image (GPT-4o dominates), $100K-300K for 10M images initial indexing
* **Estimated throughput**: 2.5-5 seconds per image (GPT-4o latency dominates)

---

## Technical Scenarios

### Scenario 1: End-to-End Ingestion Pipeline

**Description**: Process an image + generation prompt through all pipeline stages to produce a fully indexed document.

**Requirements**:
* Single GPT-4o vision call extracts all descriptions + metadata
* 4 parallel embedding calls (grouped by dimension) produce all vectors
* Document uploaded to Azure AI Search index
* Total per-image latency: 2.5-5 seconds
* Total per-image cost: ~$0.01-0.03

**Preferred Approach**: Unified GPT-4o extraction → Matryoshka embeddings at varying dimensions → batch index upload

```
Image + Prompt
    │
    ▼
GPT-4o Vision (single structured call)
    │
    ├── semantic_description → embed @ 3072 dims → semantic_vector
    ├── structural_description → embed @ 1024 dims → structural_vector
    ├── style_description → embed @ 512 dims → style_vector
    ├── characters[].semantic → embed @ 512 dims → char_N_semantic_vector
    ├── characters[].emotion → embed @ 256 dims → char_N_emotion_vector
    ├── characters[].pose → embed @ 256 dims → char_N_pose_vector
    └── metadata (scene_type, tags, etc.) → primitive fields
    │
    ▼
Azure AI Search Index Upload
```

**Implementation Details**:

```python
# Single GPT-4o extraction call with Pydantic structured output
class ImageExtraction(BaseModel):
    semantic_description: str
    structural_description: str
    style_description: str
    characters: list[CharacterDescription]
    metadata: ImageMetadata

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": f"Generation prompt: {prompt}"},
            {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}}
        ]}
    ],
    response_format=ImageExtraction,
    temperature=0.2
)
extraction: ImageExtraction = response.choices[0].message.parsed
```

```python
# Parallel embedding generation (4 calls grouped by dimension)
async def generate_all_vectors(extraction: ImageExtraction) -> dict:
    batch_3072 = [extraction.semantic_description]
    batch_1024 = [extraction.structural_description]
    batch_512 = [extraction.style_description] + [c.semantic for c in extraction.characters[:3]]
    batch_256 = [c.emotion for c in extraction.characters[:3]] + [c.pose for c in extraction.characters[:3]]

    r3072, r1024, r512, r256 = await asyncio.gather(
        embed_batch(batch_3072, dimensions=3072),
        embed_batch(batch_1024, dimensions=1024),
        embed_batch(batch_512, dimensions=512),
        embed_batch(batch_256, dimensions=256),
    )
    return assemble_document(r3072, r1024, r512, r256)
```

#### Considered Alternatives

* **Separate model calls (DINOv2 + Style Encoder + SigLIP)**: Higher structural/style quality but requires custom model deployment on Azure AI Foundry. Rejected for initial implementation due to operational complexity. Future upgrade path documented.
* **Multiple GPT-4o calls (one per description type)**: Redundant image token charges, 4x latency. Rejected in favor of single structured call.
* **`azure-ai-inference` SDK**: Less mature for OpenAI models. Rejected in favor of `openai` package with `AzureOpenAI` client.

### Scenario 2: Hybrid Retrieval Query

**Description**: User submits a text or image query, system returns ranked, diverse results.

**Requirements**:
* P95 latency < 300ms
* Hybrid BM25 + multi-vector search with configurable weights
* Application-level re-ranking (emotional, narrative, object, low-light)
* Result diversity via MMR

**Preferred Approach**: Three-stage pipeline — Azure AI Search (RRF) → Rule-based re-rank → MMR diversity

**Implementation Details**:

```python
# Stage 1: Azure AI Search hybrid query
results = search_client.search(
    search_text=query_text,
    vector_queries=[
        VectorizedQuery(vector=semantic_emb, k_nearest_neighbors=100,
                       fields="semantic_vector", weight=5.0),
        VectorizedQuery(vector=structural_emb, k_nearest_neighbors=100,
                       fields="structural_vector", weight=2.0),
        VectorizedQuery(vector=style_emb, k_nearest_neighbors=100,
                       fields="style_vector", weight=2.0),
    ],
    top=200,
    select=["image_id", "generation_prompt", "scene_type",
            "emotional_polarity", "tags", "low_light_score"],
)

# Stage 2: Application re-ranking (emotional/narrative/object/low-light)
reranked = rerank_candidates(list(results), query_context, top_n=50)

# Stage 3: MMR diversity
final = mmr_rerank(query_embedding, candidate_embeddings, scores, lambda_param=0.6, top_n=20)
```

#### Considered Alternatives

* **LLM re-ranking in real-time**: 500ms+ latency per call, exceeds 300ms budget. Reserved for future premium tier.
* **Azure AI Search semantic ranker only**: Cross-encoder re-ranking but limited to text fields, does not cover emotional/narrative/object domain-specific criteria.
* **Separate API calls per vector field**: Unnecessary — Azure AI Search parallelizes internally within a single request.

### Scenario 3: Azure AI Search Index Schema

**Description**: Define the multi-vector index schema supporting all required fields.

**Requirements**:
* Primitive fields (key, searchable, filterable, facetable, sortable)
* 3 primary vector fields (3072, 1024, 512 dims)
* 9 character sub-vector fields (3 slots × 3 types)
* HNSW with cosine similarity
* Semantic ranker configuration

**Preferred Approach**: Flattened character vectors + shared HNSW profile + semantic config

```python
# All vector fields reference a shared HNSW profile
vector_search = VectorSearch(
    algorithms=[
        HnswAlgorithmConfiguration(
            name="hnsw-cosine",
            parameters=HnswParameters(m=4, ef_construction=400, ef_search=500,
                                     metric=VectorSearchAlgorithmMetric.COSINE),
        ),
    ],
    profiles=[
        VectorSearchProfile(name="hnsw-cosine-profile",
                          algorithm_configuration_name="hnsw-cosine"),
    ],
)

# Fields include: image_id, generation_prompt, scene_type, lighting_condition,
# tags, emotional_polarity, low_light_score, character_count,
# semantic_vector (3072), structural_vector (1024), style_vector (512),
# char_0/1/2_semantic_vector (512), char_0/1/2_emotion_vector (256),
# char_0/1/2_pose_vector (256)
# Total: ~15 vector fields
```

#### Considered Alternatives

* **Nested character vectors** (`Collection(Edm.ComplexType)`): Not supported for vector search by Azure AI Search. Rejected.
* **Separate character index**: More flexible but adds cross-index join complexity. Considered for future if per-character search becomes critical.
* **Aggregated character vectors** (mean-pooled): Loses individual character identity. Rejected.

---

## Selected Approach Summary

### Architecture: GPT-4o + text-embedding-3-large + Azure AI Search

All components use Azure AI Foundry as the sole model platform:

1. **Ingestion**: Image + prompt → single GPT-4o structured vision call → metadata + descriptions
2. **Embeddings**: text-embedding-3-large with Matryoshka dimensions (3072/1024/512/256)
3. **Indexing**: Azure AI Search with flattened multi-vector schema, HNSW cosine, batch upload
4. **Retrieval**: Hybrid RRF (BM25 + 3 vector queries) → rule-based re-rank → MMR diversity
5. **Project**: Python with UV, src-layout, pydantic-settings + config.yaml, hatchling build

### Rationale

* Satisfies the **Azure AI Foundry-only constraint** without custom model deployments
* Single GPT-4o call per image minimizes latency and cost
* Matryoshka embeddings from one model replace 3+ separate embedding models
* Three-stage retrieval meets P95 < 300ms with room to spare
* Flattened character vectors work within Azure AI Search constraints
* UV + hatchling + src-layout is the modern Python standard

### Implementation Priority

1. Scaffold project (UV init, pyproject.toml, directory structure)
2. Configuration management (config.py with pydantic-settings + YAML)
3. Azure AI Foundry client (openai SDK initialization, async support)
4. Extraction module (GPT-4o vision structured call)
5. Embedding module (multi-dimension batch embeddings)
6. Index schema creation (Azure AI Search SDK)
7. Document indexer (batch upload with retry)
8. Retrieval service (hybrid search + re-rank + MMR)
9. CLI entry points
10. Tests (unit + integration)
