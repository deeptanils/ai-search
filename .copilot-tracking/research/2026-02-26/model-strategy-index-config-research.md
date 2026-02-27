<!-- markdownlint-disable-file -->
# Multimodal / Image Embedding Model Research for Azure AI Foundry

Research into multimodal and image embedding models available on Azure AI Foundry for the AI Search pipeline, addressing the need for direct image-to-vector embeddings alongside the existing text-based embedding architecture.

## Current Architecture

The pipeline currently operates in two stages:

1. **GPT-4o vision** extracts structured text descriptions from images (semantic, structural, style, character).
2. **text-embedding-3-large** embeds those text descriptions into vectors at varying Matryoshka dimensions (3072/1024/512/256).

All models are accessed through the `AzureOpenAI` client from the `openai` SDK. No direct image-to-vector path exists — all visual information is mediated through GPT-4o text output.

## Problem Statement

The current text-mediated embedding approach loses visual information that cannot be captured in text descriptions:
- Fine-grained texture, exact color distributions, spatial frequency patterns
- Subtle compositional nuances that resist verbal description
- Direct image-to-image similarity (query with an image, find visually similar images)

The user wants to add **direct image embeddings** (image → vector) so the system supports:
- Image-to-image similarity search
- Cross-modal search (text query → image results, image query → image results)
- Richer visual representation that complements text-based embeddings

---

## Q1: Multimodal / Image Embedding Models on Azure AI Foundry

### Available Models Summary

| Model | Modality | Dimensions | Azure Availability | SDK / Client | Image Input | Notes |
|-------|----------|------------|-------------------|-------------|-------------|-------|
| **text-embedding-3-large** | Text only | 256–3072 (Matryoshka) | Azure OpenAI (Foundry) | `openai` / `AzureOpenAI` | No | Current pipeline model. Text only — does NOT accept image inputs. |
| **text-embedding-3-small** | Text only | 512–1536 (Matryoshka) | Azure OpenAI (Foundry) | `openai` / `AzureOpenAI` | No | Smaller/cheaper variant. Text only. |
| **Cohere embed-english-v3.0** | Text only | 1024 | Azure AI Foundry (MaaS) | `azure-ai-inference` / Cohere SDK | No | Serverless API deployment. Text only. |
| **Cohere embed-multilingual-v3.0** | Text only | 1024 | Azure AI Foundry (MaaS) | `azure-ai-inference` / Cohere SDK | No | Serverless API deployment. Text only. |
| **Cohere Embed v4** | Multimodal (text + image) | 256–1024 (configurable) | Azure AI Foundry (MaaS) — check region availability | `azure-ai-inference` / Cohere SDK | Yes | Native multimodal. See Q3 for details. |
| **Azure Computer Vision 4.0 (Florence)** | Multimodal (text + image) | 1024 (fixed) | Azure Cognitive Services (separate resource) | `azure-ai-vision-imageanalysis` / REST | Yes | Not in Foundry model catalog. Separate service. See Q2. |
| **GPT-4o** | Chat/Vision (generation, not embedding) | N/A | Azure OpenAI (Foundry) | `openai` / `AzureOpenAI` | Yes (for chat) | Does NOT have an embeddings endpoint. Cannot produce vector embeddings directly from images. |
| **CLIP / SigLIP / EVA-CLIP** | Multimodal | 512–1024 | NOT available as managed | Custom container (Azure ML) | Yes | Require custom model deployment. Not a managed Foundry offering. |
| **Meta Llama embeddings** | Text only | Various | Azure AI Foundry (MaaS) | `azure-ai-inference` | No | Text only. |
| **Mistral Embed** | Text only | 1024 | Azure AI Foundry (MaaS) | `azure-ai-inference` | No | Text only. |

### Key Finding

Only **two viable options** exist for image-to-vector embeddings with managed Azure deployments (no custom containers):

1. **Azure Computer Vision 4.0 (Florence)** — separate Azure service, not Foundry model catalog
2. **Cohere Embed v4** — available in Azure AI Foundry as serverless API (MaaS), native multimodal

GPT-4o does NOT expose an embedding endpoint. `text-embedding-3-large` does NOT accept image inputs.

---

## Q2: Azure Computer Vision 4.0 (Florence) Multimodal Embeddings

### Overview

Azure Computer Vision 4.0 uses the **Florence** foundation model to provide multimodal embeddings through two endpoints:

| Endpoint | Input | Output | API Path |
|----------|-------|--------|----------|
| `vectorizeImage` | Image (URL or binary) | 1024-dim float vector | `POST /computervision/retrieval:vectorizeImage` |
| `vectorizeText` | Text string | 1024-dim float vector | `POST /computervision/retrieval:vectorizeText` |

### Characteristics

- **Shared embedding space**: Image vectors and text vectors live in the same 1024-dimensional space. Cosine similarity works cross-modally.
- **Fixed dimensions**: 1024 only — no Matryoshka support, no configurable dimensions.
- **API version**: `2024-02-01` (stable).
- **Region availability**: East US, West US, West Europe, Southeast Asia, Japan East, Australia East (check for updates — not all regions supported).
- **Authentication**: Azure Cognitive Services key or Entra ID token.
- **Rate limits**: Up to 1000 requests/minute (varies by tier).

### SDK and Client

```python
# Option A: REST API (recommended for image embeddings)
import httpx

endpoint = "https://<resource-name>.cognitiveservices.azure.com"
api_version = "2024-02-01"

# Image → vector
response = httpx.post(
    f"{endpoint}/computervision/retrieval:vectorizeImage",
    params={"api-version": api_version, "model-version": "2023-04-15"},
    headers={"Ocp-Apim-Subscription-Key": api_key},
    json={"url": image_url},  # or send binary with content-type image/*
)
image_vector = response.json()["vector"]  # list[float], len=1024

# Text → vector (same embedding space)
response = httpx.post(
    f"{endpoint}/computervision/retrieval:vectorizeText",
    params={"api-version": api_version, "model-version": "2023-04-15"},
    headers={"Ocp-Apim-Subscription-Key": api_key},
    json={"text": "a woman in a red dress standing in rain"},
)
text_vector = response.json()["vector"]  # list[float], len=1024
```

```python
# Option B: azure-ai-vision-imageanalysis SDK
# Note: The SDK primarily focuses on image analysis (captions, tags, etc.)
# Multimodal embeddings (vectorize) are best accessed via REST API
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.core.credentials import AzureKeyCredential
```

### Relationship to Azure AI Foundry

- Azure Computer Vision 4.0 is a **separate Azure resource** (Cognitive Services / AI Services multi-service resource).
- It is NOT deployed through the Azure AI Foundry model catalog.
- However, it CAN be called from code running alongside Foundry models — just requires provisioning a Computer Vision resource separately.
- If the constraint is "Azure AI Foundry managed deployments ONLY" (no separate Azure services), then Florence is NOT an option.
- If the constraint is "Azure cloud services" (broader), Florence IS available.

### Strengths and Limitations

| Strength | Limitation |
|----------|-----------|
| Same embedding space for image + text | Fixed 1024 dimensions (no Matryoshka) |
| Fast inference (~50-100ms per image) | Separate Azure resource provisioning |
| Production-grade, Microsoft-managed | Different SDK/auth from OpenAI client |
| No custom model deployment needed | Limited model versioning (no fine-tuning) |
| Stable GA API | Region availability restrictions |

---

## Q3: Cohere Embed v4 on Azure AI Foundry

### Overview

Cohere Embed v4 is a **native multimodal embedding model** that accepts both text and image inputs and produces vectors in a shared embedding space.

### Azure AI Foundry Availability

- Available as a **serverless API deployment** (Models as a Service / MaaS) in the Azure AI Foundry model catalog.
- Deployment name example: `cohere-embed-v4-0` (check catalog for exact name).
- Billing: Pay-per-token, billed through Azure AI Foundry.
- Region: Check Azure AI Foundry model catalog for supported regions (typically East US, West Europe, Sweden Central).

### Capabilities

| Feature | Detail |
|---------|--------|
| **Text input** | Yes — accepts text strings |
| **Image input** | Yes — accepts base64-encoded images or image URLs |
| **Shared embedding space** | Yes — image and text vectors are comparable via cosine similarity |
| **Configurable dimensions** | 256, 384, 512, 768, 1024 |
| **Input types** | `search_document`, `search_query`, `classification`, `clustering` |
| **Max tokens** | 512 tokens per text input |
| **Max images** | Varies — check model documentation |
| **Compression** | Supports int8 and binary quantization natively |

### SDK Usage

Cohere models on Azure AI Foundry use the `azure-ai-inference` SDK or the Cohere SDK — they are NOT compatible with the `openai` / `AzureOpenAI` client.

```python
# Option A: azure-ai-inference SDK
from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential

client = EmbeddingsClient(
    endpoint="https://<deployment-name>.<region>.models.ai.azure.com",
    credential=AzureKeyCredential(api_key),
)

# Text embedding
response = client.embed(
    input=["a dramatic scene with rain and neon lights"],
    input_type="search_query",
    dimensions=1024,
)
text_vector = response.data[0].embedding  # list[float], len=1024

# Image embedding (base64)
import base64
with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = client.embed(
    input=[{"image": image_b64}],  # image input format
    input_type="search_document",
    dimensions=1024,
)
image_vector = response.data[0].embedding  # list[float], len=1024
```

```python
# Option B: Cohere Python SDK
import cohere

co = cohere.ClientV2(
    api_key=api_key,
    base_url="https://<deployment-name>.<region>.models.ai.azure.com/v2",
)

response = co.embed(
    texts=["query text"],
    images=["data:image/jpeg;base64,..."],  # or image URL
    model="cohere-embed-v4-0",
    input_type="search_query",
    embedding_types=["float"],
    output_dimension=1024,
)
```

### Strengths and Limitations

| Strength | Limitation |
|----------|-----------|
| Native multimodal (text + image) | Different SDK from current `openai` client |
| Azure AI Foundry managed deployment | Not OpenAI-API compatible |
| Configurable dimensions (256–1024) | Max 1024 dims (vs text-embedding-3-large's 3072) |
| Supports int8/binary quantization | Pay-per-token cost on top of Azure OpenAI |
| Same embedding space for cross-modal | Adds `azure-ai-inference` or `cohere` dependency |
| True managed — no separate resource | May not be GA in all regions |

---

## Q4: Architecture Recommendations

### Option A: Hybrid — Keep text-embedding-3-large + Add Azure Computer Vision Florence

**Architecture**: Two embedding systems running side by side.

```
Image
  │
  ├──► GPT-4o extraction → text descriptions → text-embedding-3-large
  │        (semantic, structural, style, character vectors)
  │
  └──► Azure Computer Vision 4.0 (Florence) → image_vector (1024 dims)
           (direct image-to-vector, shared space with vectorizeText)
```

- **Text-based search**: Uses existing multi-vector fields (semantic, structural, style, character) from text-embedding-3-large.
- **Image similarity search**: Uses new `image_vector` field from Florence. Query with image (vectorizeImage) or text (vectorizeText).
- **Pros**: Minimal disruption to existing pipeline. Best-of-both-worlds: rich text semantics + direct visual similarity.
- **Cons**: Requires separate Azure Computer Vision resource. Two different embedding spaces. Adds complexity.

### Option B: Replace text-embedding-3-large with Cohere Embed v4

**Architecture**: Single multimodal model for all embeddings.

```
Image
  │
  ├──► GPT-4o extraction → text descriptions → Cohere Embed v4 (text mode)
  │        (semantic, structural, style, character vectors at varying dims)
  │
  └──► Cohere Embed v4 (image mode) → image_vector (1024 dims)
           (direct image-to-vector, same embedding space as text)
```

- **Pros**: Single model, single embedding space. Cross-modal similarity out of the box. Stays within Azure AI Foundry.
- **Cons**: Max 1024 dims (lose the 3072-dim semantic vector). Different SDK. Replaces a proven model. No Matryoshka support equivalent.

### Option C: Keep text-embedding-3-large + Add Cohere Embed v4 for Image Embeddings

**Architecture**: Two models — text-embedding-3-large for text, Cohere Embed v4 for images only.

```
Image
  │
  ├──► GPT-4o extraction → text descriptions → text-embedding-3-large
  │        (semantic @3072, structural @1024, style @512, character @512/256)
  │
  └──► Cohere Embed v4 (image mode) → image_vector (1024 dims)
  └──► Cohere Embed v4 (text mode) → image_text_vector (1024 dims)
           (for cross-modal query: text → same space as image vectors)
```

- **Pros**: Preserves existing multi-vector richness. Adds image similarity. Stays within Azure AI Foundry. Cohere handles cross-modal queries.
- **Cons**: Two embedding spaces (text-embedding-3-large space ≠ Cohere space). Need Cohere for query-time text embedding when doing image similarity. Adds dependency.

### Option D: Keep text-embedding-3-large + Add Azure Computer Vision Florence (RECOMMENDED)

Same as Option A but with clearer justification:

**Architecture**: text-embedding-3-large for rich text semantics + Florence for direct visual similarity.

```
Image + Prompt
  │
  ├── GPT-4o Vision (structured extraction)
  │     ├── semantic_description → text-embedding-3-large @ 3072 → semantic_vector
  │     ├── structural_description → text-embedding-3-large @ 1024 → structural_vector
  │     ├── style_description → text-embedding-3-large @ 512 → style_vector
  │     └── characters[].* → text-embedding-3-large @ 512/256 → char_N_*_vector
  │
  └── Azure Computer Vision 4.0 (Florence)
        └── image_url / image_bytes → vectorizeImage → image_vector (1024)
```

**For queries**:

```
Text Query → text-embedding-3-large → semantic/structural/style vectors (text space)
           → Florence vectorizeText → image_query_vector (Florence space)

Image Query → Florence vectorizeImage → image_query_vector (Florence space)
```

### Recommended Option: D (Hybrid text-embedding-3-large + Florence)

**Rationale**:

1. **Preserves existing architecture**: All text embedding code, Matryoshka dimensions, and multi-vector search remain unchanged.
2. **Adds genuine image similarity**: Florence `vectorizeImage` provides direct image-to-vector with no text mediation.
3. **Cross-modal via Florence**: Text queries can also use `vectorizeText` to search in the image embedding space — single shared 1024-dim space.
4. **Minimal code changes**: Add one new embedding module, one new vector field, one new config section.
5. **Production-grade**: Both text-embedding-3-large and Florence are GA, Microsoft-managed services.
6. **Two complementary search strategies**: Text-based multi-vector (rich semantics, 5+ vector fields) AND image-based visual similarity (1 vector field, direct visual matching).

**Trade-off acknowledged**: Florence is a separate Azure resource (not in the Foundry model catalog). If the user's constraint is strictly "Azure AI Foundry model catalog deployments only," then Option C (Cohere Embed v4) is the alternative. Both are valid.

---

## Recommended Model Set

### Models to Deploy

| Model | Deployment Target | Purpose | SDK |
|-------|-------------------|---------|-----|
| **GPT-4o** | Azure AI Foundry (OpenAI) | Vision extraction + query expansion | `openai` / `AzureOpenAI` |
| **text-embedding-3-large** | Azure AI Foundry (OpenAI) | Text embeddings (semantic, structural, style, character) at 256–3072 dims | `openai` / `AzureOpenAI` |
| **Azure Computer Vision 4.0** | Azure Cognitive Services (separate resource) | Image → 1024-dim vector, Text → 1024-dim vector (shared space) | `httpx` REST or `azure-ai-vision-imageanalysis` |

### Alternative Model Set (Foundry-Only Constraint)

| Model | Deployment Target | Purpose | SDK |
|-------|-------------------|---------|-----|
| **GPT-4o** | Azure AI Foundry (OpenAI) | Vision extraction + query expansion | `openai` / `AzureOpenAI` |
| **text-embedding-3-large** | Azure AI Foundry (OpenAI) | Text embeddings at 256–3072 dims | `openai` / `AzureOpenAI` |
| **Cohere Embed v4** | Azure AI Foundry (MaaS serverless) | Image embeddings + cross-modal text embeddings at 1024 dims | `azure-ai-inference` |

---

## Config Changes Needed

### config.yaml Additions

```yaml
models:
  embedding_model: text-embedding-3-large
  llm_model: gpt-4o
  image_embedding_model: azure-cv-florence  # or "cohere-embed-v4-0"

index:
  vector_dimensions:
    semantic: 3072
    structural: 1024
    style: 512
    character_semantic: 512
    character_emotion: 256
    character_pose: 256
    image: 1024  # NEW — direct image embedding dimension
```

### .env Additions (for Florence)

```env
# Azure Computer Vision (Florence image embeddings)
AZURE_CV_ENDPOINT=https://<resource>.cognitiveservices.azure.com
AZURE_CV_API_KEY=<key>
```

### .env Additions (for Cohere, if chosen)

```env
# Cohere Embed v4 on Azure AI Foundry
COHERE_EMBED_ENDPOINT=https://<deployment>.<region>.models.ai.azure.com
COHERE_EMBED_API_KEY=<key>
```

---

## Codebase Changes Needed

### 1. Config Layer

- **config.py**: Add `image_embedding_model` to `ModelsConfig`. Add `image: int = 1024` to `VectorDimensionsConfig`. Add new secrets model for the image embedding provider (Florence or Cohere).
- **config.yaml**: Add `image_embedding_model` and `image` dimension.
- **.env**: Add image embedding service credentials.

### 2. Client Layer

- **clients.py**: Add factory function for the image embedding client.
  - Florence: `get_cv_client()` returning an `httpx.AsyncClient` configured with the CV endpoint.
  - Cohere: `get_cohere_embed_client()` returning an `EmbeddingsClient` from `azure-ai-inference`.

### 3. Embedding Layer

- **embeddings/image.py** (NEW): Module for direct image-to-vector embedding.
  - `embed_image(image_url_or_bytes) → list[float]` — calls Florence `vectorizeImage` or Cohere `embed(image)`.
  - `embed_text_for_image_search(text) → list[float]` — calls Florence `vectorizeText` or Cohere `embed(text, input_type="search_query")` to produce vectors in the image embedding space.
- **embeddings/pipeline.py**: Add `embed_image` call alongside existing text embeddings.

### 4. Indexing Layer

- **indexing/schema.py**: Add `image_vector` field (1024 dims) to the index schema.
- **models.py**: Add `image_vector: list[float]` to `SearchDocument` and `ImageVectors`.

### 5. Retrieval Layer

- **retrieval/query.py**: If query includes an image, call `embed_image()`. If text-only query, optionally call `embed_text_for_image_search()` for image space search.
- **retrieval/search.py**: Add `image_vector` to the vector queries with a configurable weight.
- **config.yaml**: Add `image_weight` to search weights.

### 6. Dependencies

- **pyproject.toml**: Add `azure-ai-vision-imageanalysis>=1.0.0` (for Florence) OR `azure-ai-inference>=1.0.0` (for Cohere).

---

## Impact Analysis

### Storage Impact

Adding one `image_vector` field at 1024 dimensions per document:

| Metric | Current | With image_vector | Delta |
|--------|---------|-------------------|-------|
| Vectors per document | 7,680 dims (~30 KB) | 8,704 dims (~34 KB) | +1,024 dims (+4 KB) |
| 10M documents | ~280 GB | ~317 GB | +37 GB |
| API calls per image | 5 (1 GPT-4o + 4 embed) | 6 (1 GPT-4o + 4 embed + 1 image embed) | +1 |

### Latency Impact

Florence `vectorizeImage` runs in ~50-100ms and can execute in parallel with the GPT-4o call (it doesn't depend on extraction output). Net latency impact: ~0ms (fully parallelizable).

### Cost Impact

- Florence: ~$1 per 1,000 transactions (check current pricing).
- At 10M images: ~$10,000 additional for image embeddings.
- Compared to GPT-4o extraction cost ($100K-300K), this is minor (~3-5% increase).

---

## Decision Matrix

| Criteria | Florence (Option D) | Cohere v4 (Option C) | Score Weight |
|----------|:-------------------:|:--------------------:|:-----------:|
| Stays in Azure AI Foundry catalog | No (separate resource) | Yes | 0.15 |
| Production readiness (GA) | GA | GA (verify) | 0.15 |
| Shared image-text embedding space | Yes (1024-dim) | Yes (up to 1024-dim) | 0.20 |
| Minimal codebase disruption | High (add 1 module) | Medium (add 1 module + new SDK) | 0.15 |
| Embedding quality for images | High (Florence purpose-built for vision) | High (Cohere multimodal) | 0.20 |
| SDK consistency with existing code | Moderate (REST/httpx, not openai) | Low (azure-ai-inference, not openai) | 0.10 |
| Cost | Low (~$1/1K images) | Medium (pay-per-token MaaS) | 0.05 |
| **Weighted Score** | **Higher** | **Slightly Lower** | — |

**Verdict**: Florence (Option D) is the primary recommendation. It is a purpose-built vision embedding model with a proven track record, straightforward REST API, and minimal integration overhead. Cohere Embed v4 (Option C) is the recommended alternative if strict Foundry-catalog-only constraint applies.

---

## Summary of Findings

1. **text-embedding-3-large does NOT accept image inputs** — text only, confirmed.
2. **GPT-4o does NOT have an embedding endpoint** — it is a chat/completion model, not an embedding model.
3. **No CLIP/SigLIP/EVA-CLIP available as managed deployments** on Azure AI Foundry — these require custom container deployment via Azure ML.
4. **Two viable managed options exist**:
   - **Azure Computer Vision 4.0 (Florence)**: Separate Azure resource, 1024-dim shared image-text space, REST API.
   - **Cohere Embed v4**: Azure AI Foundry MaaS, up to 1024-dim shared image-text space, `azure-ai-inference` SDK.
5. **Recommended approach**: Add Florence as a parallel image embedding path alongside the existing text-embedding-3-large pipeline. This preserves all current multi-vector richness while adding direct visual similarity search.
6. **Config impact**: Add `image_embedding_model`, `image` dimension, image embedding service secrets, and `image_weight` search weight.
7. **Code impact**: ~4 files modified, ~1 new file (`embeddings/image.py`), ~1 new dependency.
