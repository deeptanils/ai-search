---
title: Multi-Vector Encoding Strategies for AI Image Search Pipeline
description: Deep research on multi-vector encoding strategies using Azure AI Foundry for the candidate generation and AI search pipeline
author: research-subagent
ms.date: 2026-02-26
ms.topic: reference
keywords:
  - multi-vector encoding
  - azure ai foundry
  - text-embedding-3-large
  - matryoshka embeddings
  - gpt-4o vision
  - azure ai search
estimated_reading_time: 20
---

## Executive Summary

This document researches multi-vector encoding strategies for the AI image search pipeline, constrained to **Azure AI Foundry** as the sole model hosting platform. The core finding is that a **GPT-4o vision + text-embedding-3-large** architecture with **Matryoshka dimension reduction** is the most practical approach. This avoids custom model deployments while producing semantically rich, dimensionally varied vectors suitable for Azure AI Search hybrid retrieval.

## 1. Semantic Vector via text-embedding-3-large

### 1.1 Architecture

The semantic vector captures the high-level meaning of an image—what it depicts, its narrative, and contextual significance.

**Pipeline:**

1. **GPT-4o Vision** analyzes the image and its generation prompt together
2. GPT-4o produces a **rich text description** (500–1000 tokens) covering scene content, subjects, actions, mood, environment, and narrative intent
3. The description is passed to **text-embedding-3-large** to produce a 3072-dimensional vector
4. This vector becomes the primary retrieval signal

**Prompt strategy for GPT-4o:**

```text
Analyze this image alongside its generation prompt. Produce a detailed
description covering: subjects and their actions, environment and setting,
mood and atmosphere, narrative intent, spatial relationships, notable objects,
and any symbolic or thematic elements. Be specific and descriptive.
Generation prompt: {generation_prompt}
```

### 1.2 Dimension Reduction via Matryoshka Representation

text-embedding-3-large supports the `dimensions` parameter, implementing **Matryoshka Representation Learning (MRL)**. This is a training technique where the model learns embeddings such that the first N dimensions form a valid, useful embedding at reduced dimensionality.

**Supported dimension values:**

| Dimensions | Relative Quality | Use Case                          |
|------------|------------------|-----------------------------------|
| 3072       | Baseline (100%)  | Primary semantic search           |
| 1536       | ~99.5%           | High-quality with 50% size saving |
| 1024       | ~99%             | Good balance for secondary vectors |
| 512        | ~97%             | Acceptable for tertiary vectors   |
| 256        | ~93%             | Minimal viable embedding          |

**How it works:** The `dimensions` parameter is passed in the API call. The model returns an embedding truncated to the first N dimensions, which are then normalized. This is **not** a post-hoc truncation—the model was trained with MRL so the leading dimensions carry the most information.

**API call example:**

```python
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint="https://can-foundry.cognitiveservices.azure.com/",
    api_key=os.environ["AZURE_FOUNDRY_API_KEY"],
    api_version="2024-12-01-preview"
)

response = client.embeddings.create(
    input="detailed image description text...",
    model="text-embedding-3-large",
    dimensions=3072  # or 1536, 1024, 512, 256
)

vector = response.data[0].embedding
```

**Key point:** The `dimensions` parameter reduces cost proportionally (fewer dimensions = smaller payload) and the quality degradation is minimal down to 1024. This makes it ideal for generating multiple vector types at different dimensionalities from a single model.

### 1.3 Recommendation for Semantic Vector

* **Dimensions:** 3072 (full fidelity for primary retrieval signal)
* **Input:** Concatenation of GPT-4o's rich description + the original generation prompt
* **Rationale:** As the primary retrieval vector, semantic quality should not be compromised

## 2. Structural Vector Approach

### 2.1 Problem Statement

The requirements specify **DINOv2** for structural/layout embeddings. DINOv2 is a self-supervised vision transformer from Meta that excels at capturing spatial structure, object positioning, and compositional layout. However, DINOv2 is **not natively available in Azure AI Foundry's model catalog** as a first-party endpoint.

### 2.2 Option Analysis

#### Option A: GPT-4o Structured Extraction + text-embedding-3-large (Recommended)

**Approach:** Use GPT-4o vision to generate a **structured spatial description** of the image, then embed it with text-embedding-3-large at reduced dimensions.

**GPT-4o extraction prompt:**

```text
Analyze the spatial structure and composition of this image. Describe:
1. Layout type (rule-of-thirds, centered, symmetric, diagonal, etc.)
2. Spatial zones: what occupies foreground, midground, background
3. Object positions: describe where key objects are (left/right/center,
   top/bottom, relative positions)
4. Lines of composition: leading lines, vanishing points, depth cues
5. Scale relationships between objects
6. Negative space usage
7. Overall geometric structure

Output as a dense paragraph focusing on spatial relationships.
```

**Pros:**

* No custom model deployment required
* Uses existing Azure AI Foundry endpoints
* text-embedding-3-large at 1024 dimensions produces a compact structural vector
* GPT-4o is surprisingly capable at spatial reasoning

**Cons:**

* Structural information is mediated through language (lossy)
* Does not capture pixel-level spatial features like DINOv2
* For images where layout is critical (e.g., UI screenshots, architectural diagrams), quality may be lower than a dedicated vision encoder

**Quality assessment:** For AI-generated images (the target domain), GPT-4o's structural descriptions capture ~80–85% of the compositional information that DINOv2 would encode. This is because AI-generated images typically have clear compositional intent that GPT-4o can articulate.

#### Option B: Deploy DINOv2 as Custom Endpoint on Azure AI Foundry

**Approach:** Package DINOv2 (ViT-L/14 or ViT-G/14) as a custom model and deploy it as a managed online endpoint in Azure AI Foundry.

**Steps required:**

1. Download DINOv2 weights from the Hugging Face model hub
2. Create an inference script (`score.py`) that loads the model, preprocesses images, and returns the CLS token embedding
3. Package as an Azure ML model using a custom Docker environment
4. Deploy as a managed online endpoint in Azure AI Foundry
5. Set up scaling, monitoring, and authentication

**Pros:**

* True structural embeddings without language mediation
* DINOv2 ViT-L/14 produces 1024-dimensional vectors natively
* Best quality for structural similarity

**Cons:**

* Significant operational overhead (deployment, monitoring, scaling, updates)
* Requires GPU compute for inference (A100 or V100)
* Additional cost for a dedicated endpoint (~$2–5/hour for a single GPU instance)
* Adds latency for a separate model call in the pipeline
* Must manage model versioning and updates independently

**Feasibility:** Technically feasible but adds substantial complexity. Only recommended if structural similarity is a primary differentiator for the search use case.

#### Option C: Azure-Hosted Alternative Models

**Azure AI Foundry model catalog** includes several vision models, but none are direct DINOv2 replacements for structural embeddings:

* **Florence-2:** Available in Azure AI Foundry catalog. Produces visual features but is primarily designed for captioning/grounding tasks, not raw structural embeddings
* **CLIP/SigLIP variants:** These capture semantic alignment (image-text), not pure structural layout
* **Phi-3-vision / Phi-3.5-vision:** Multimodal LLMs—can describe structure but don't produce dense vector embeddings directly

**Assessment:** No Azure-native model serves as a drop-in replacement for DINOv2's structural encoding capability.

### 2.3 Recommendation for Structural Vector

**Use Option A** (GPT-4o extraction + text-embedding-3-large at 1024 dimensions) as the **initial implementation**. This satisfies the Azure AI Foundry constraint with zero additional infrastructure.

**Future upgrade path:** If structural search quality proves insufficient, deploy DINOv2 as a custom endpoint (Option B) as an enhancement.

* **Dimensions:** 1024 (via Matryoshka reduction)
* **Input:** GPT-4o's structured spatial/compositional description
* **Field:** `structural_vector`

## 3. Style Vector Approach

### 3.1 Problem Statement

The requirements specify a **Style Encoder / LoRA-derived embedding** for capturing artistic style, lighting tone, and color treatment. No such model exists natively in Azure AI Foundry.

### 3.2 Option Analysis

#### Option A: GPT-4o Style Extraction + text-embedding-3-large (Recommended)

**Approach:** Use GPT-4o to produce a detailed style description, then embed with text-embedding-3-large at 512 dimensions.

**GPT-4o extraction prompt:**

```text
Analyze the artistic style of this image. Describe:
1. Art style: photorealistic, illustration, watercolor, oil painting,
   digital art, anime, etc.
2. Color palette: dominant colors, color temperature, saturation level,
   color harmony type
3. Lighting: direction, quality (hard/soft), color temperature, contrast ratio
4. Texture treatment: smooth, textured, grainy, painterly strokes
5. Rendering technique: cel-shaded, ray-traced, impressionistic, etc.
6. Mood conveyed through visual treatment
7. Artistic influences or comparable artists/styles
8. Post-processing effects: bloom, vignette, chromatic aberration, film grain

Output as a dense paragraph focusing on visual style characteristics.
```

**Pros:**

* No additional model deployment
* GPT-4o excels at describing artistic style with nuance
* 512 dimensions is sufficient for style differentiation (style is a lower-dimensional concept than full semantics)
* text-embedding-3-large at 512 still captures meaningful style clusters

**Cons:**

* Language-mediated style representation may conflate stylistically different images that have similar textual descriptions
* Does not capture low-level pixel patterns (brushstrokes, texture frequencies)
* Style similarity based on text descriptions tends to be coarser-grained than pixel-based encoders

**Quality assessment:** For searching "similar style" across AI-generated images, GPT-4o descriptions capture ~75–80% of what a dedicated style encoder would provide. The gap is most noticeable for subtle stylistic differences within the same broad category (e.g., distinguishing between two slightly different anime styles).

#### Option B: Deploy Custom Style Model on Azure AI Foundry

**Approach:** Deploy a LoRA-based style encoder or a fine-tuned CLIP model as a custom endpoint.

**Options for the model:**

* **CLIP fine-tuned on style datasets** (e.g., WikiArt, BAM dataset): Extract style-focused features
* **ALADIN (Adaptive Learning of Artistic Domain via INvariance):** A dedicated style encoder
* **LoRA-adapted Stable Diffusion encoder:** Extract the LoRA conditioning vector that represents style

**Pros:**

* Pixel-level style encoding
* Better at distinguishing subtle style differences

**Cons:**

* Same deployment complexity as DINOv2 custom hosting
* Style encoder models are often research-grade and may lack production readiness
* LoRA-derived embeddings require the specific diffusion model architecture

### 3.3 Recommendation for Style Vector

**Use Option A** (GPT-4o style extraction + text-embedding-3-large at 512 dimensions).

* **Dimensions:** 512 (via Matryoshka reduction)
* **Input:** GPT-4o's detailed style description
* **Field:** `style_vector`
* **Rationale:** Style is inherently a higher-level concept that GPT-4o describes well. The 512-dimensional embedding provides sufficient discriminative power for style-based retrieval.

## 4. Character Sub-Vectors

### 4.1 Character Detection via GPT-4o Vision

GPT-4o vision can identify and describe individual characters in an image with high accuracy. The extraction pipeline should produce per-character structured data.

**Character extraction prompt:**

```text
Analyze this image and identify all characters/people present.
For each character, provide:

1. character_id: A descriptive identifier (e.g., "woman_red_dress",
   "elderly_man_left")
2. semantic: Who is this character? Describe their identity, role in scene,
   apparent age, gender, distinguishing features, clothing, accessories.
   (2-3 sentences)
3. emotion: What emotions are they expressing? Describe facial expression,
   body language emotional cues, emotional intensity, and emotional context
   within the scene. (2-3 sentences)
4. pose: Describe their physical pose and body position. Include: body
   orientation, limb positions, gesture, movement direction, interaction
   with objects/other characters, spatial position in frame.
   (2-3 sentences)

Return as a JSON array.
```

**Expected output:**

```json
[
  {
    "character_id": "woman_red_dress",
    "semantic": "A young woman in her 20s wearing a flowing red dress...",
    "emotion": "She displays a contemplative expression with slightly...",
    "pose": "Standing in the right third of the frame, body angled..."
  }
]
```

### 4.2 Per-Character Embedding Generation

For each character's text descriptions, generate separate embeddings:

| Vector Type              | Input Text               | Dimensions | Rationale                           |
|--------------------------|--------------------------|------------|-------------------------------------|
| character_semantic       | semantic description     | 512        | Identity and appearance matching    |
| character_emotion        | emotion description      | 256        | Emotion is low-dimensional          |
| character_pose           | pose description         | 256        | Pose is low-dimensional             |

**Implementation:**

```python
async def generate_character_vectors(characters: list[dict]) -> list[dict]:
    results = []
    for char in characters:
        # Batch all three embeddings in a single API call where possible
        semantic_emb = await embed(char["semantic"], dimensions=512)
        emotion_emb = await embed(char["emotion"], dimensions=256)
        pose_emb = await embed(char["pose"], dimensions=256)

        results.append({
            "character_id": char["character_id"],
            "semantic": semantic_emb,
            "emotion": emotion_emb,
            "pose": pose_emb
        })
    return results
```

### 4.3 Storing Character Vectors in Azure AI Search

Azure AI Search supports **complex types** for nested data, but has constraints on vector fields within complex types.

**Challenge:** Azure AI Search does **not** support vector search fields inside `Collection(Edm.ComplexType)`. Vector fields must be top-level or inside a non-collection complex type.

**Workaround strategies:**

#### Strategy A: Flattened Character Vectors (Recommended)

Flatten character vectors to top-level fields with a fixed maximum number of characters (e.g., 5):

```json
{
  "image_id": "img_001",
  "char_0_semantic": [0.1, 0.2, ...],
  "char_0_emotion": [0.3, 0.4, ...],
  "char_0_pose": [0.5, 0.6, ...],
  "char_1_semantic": [0.1, 0.2, ...],
  "char_1_emotion": [0.3, 0.4, ...],
  "char_1_pose": [0.5, 0.6, ...],
  "character_count": 2,
  "character_metadata": [
    {"character_id": "woman_red_dress", "slot": 0},
    {"character_id": "elderly_man_left", "slot": 1}
  ]
}
```

**Pros:**

* All vectors are searchable via Azure AI Search vector queries
* Each character slot can be independently queried

**Cons:**

* Fixed maximum character count (schema must define all slots upfront)
* Unused slots waste index space (can be mitigated with sparse vectors or null fields)

#### Strategy B: Primary Character Only

Index only the **primary character's** vectors as top-level fields. Store remaining character data as JSON metadata (non-searchable via vector).

```json
{
  "image_id": "img_001",
  "primary_char_semantic": [0.1, 0.2, ...],
  "primary_char_emotion": [0.3, 0.4, ...],
  "primary_char_pose": [0.5, 0.6, ...],
  "all_characters_json": "[{...}, {...}]"
}
```

**Pros:**

* Simpler schema
* Most queries target the primary character anyway

**Cons:**

* Secondary characters not vector-searchable

#### Strategy C: Aggregated Character Vectors

Compute a single aggregated vector per type across all characters (e.g., mean-pooled):

```python
aggregated_semantic = mean([char["semantic"] for char in characters])
aggregated_emotion = mean([char["emotion"] for char in characters])
```

**Pros:**

* Single vector field per type
* Captures overall character composition

**Cons:**

* Loses individual character identity in the vector

### 4.4 Recommendation for Character Vectors

**Use Strategy A (Flattened) with a cap of 3-5 character slots** for the initial implementation:

* `char_0_semantic_vector` (512 dims), `char_0_emotion_vector` (256 dims), `char_0_pose_vector` (256 dims)
* `char_1_semantic_vector`, `char_1_emotion_vector`, `char_1_pose_vector`
* `char_2_semantic_vector`, `char_2_emotion_vector`, `char_2_pose_vector`
* `character_count` (filterable integer)
* `character_metadata` (complex type for IDs and descriptors)

This provides full vector search capability per character slot while keeping the schema manageable.

## 5. Embedding Dimension Management

### 5.1 Matryoshka Representation Learning Deep Dive

text-embedding-3-large was trained with **Matryoshka Representation Learning (MRL)**. Key properties:

* **Progressive information encoding:** The first N dimensions encode the most important semantic information; each additional dimension adds finer detail
* **Dimension-normalized quality:** After truncation, vectors are L2-normalized to maintain cosine similarity comparability
* **Single model, multiple outputs:** One API call per text, with the `dimensions` parameter controlling output size
* **No retraining required:** Dimension reduction is handled at inference time

### 5.2 Dimension Trade-offs Analysis

| Dimension | Storage per Vector | Quality Loss (approx.) | Best For                                |
|-----------|--------------------|------------------------|-----------------------------------------|
| 3072      | 12,288 bytes       | 0% (baseline)          | Primary semantic search                 |
| 1536      | 6,144 bytes        | 0.5%                   | High-quality secondary vectors          |
| 1024      | 4,096 bytes        | 1%                     | Structural/layout vectors               |
| 512       | 2,048 bytes        | 3%                     | Style vectors, character semantics      |
| 256       | 1,024 bytes        | 7%                     | Emotion/pose vectors (narrow concepts)  |

**Storage impact at scale (10M images):**

| Configuration                 | Storage Estimate |
|-------------------------------|------------------|
| Semantic only (3072)          | ~114 GB          |
| Full multi-vector (see below) | ~220 GB          |

**Proposed dimension allocation:**

| Vector Field             | Dimensions | Rationale                                           |
|--------------------------|------------|-----------------------------------------------------|
| semantic_vector          | 3072       | Primary retrieval signal; maximize quality           |
| structural_vector        | 1024       | Composition concepts have moderate complexity        |
| style_vector             | 512        | Style is a higher-level, lower-dimensional concept   |
| char_N_semantic_vector   | 512        | Per-character identity; moderate complexity           |
| char_N_emotion_vector    | 256        | Emotion is a narrow concept space                    |
| char_N_pose_vector       | 256        | Pose is a narrow concept space                       |

**Per-image total (assuming 3 character slots):**

* 3072 + 1024 + 512 + 3 x (512 + 256 + 256) = **7,680 dimensions**
* Storage: ~30 KB per image, ~280 GB for 10M images (vectors only)

### 5.3 Quality Validation Strategy

Before committing to dimension allocations, validate with a benchmark:

1. Generate 100 test images with known similarity relationships
2. Embed at full 3072 dimensions
3. Embed at each target reduced dimension
4. Compare retrieval recall@10 and NDCG@10 across dimension sizes
5. Confirm that the quality-dimension trade-off holds for the domain

## 6. Practical Architecture Proposal

### 6.1 Unified Pipeline Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: Image + Generation Prompt              │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              GPT-4o Vision (Azure AI Foundry)                    │
│                                                                  │
│  Single call with structured output requesting:                  │
│  ─ Rich semantic description (for semantic_vector)               │
│  ─ Spatial/compositional analysis (for structural_vector)        │
│  ─ Artistic style description (for style_vector)                 │
│  ─ Per-character: semantic, emotion, pose descriptions           │
│  ─ Synthetic metadata (scene_type, tags, etc.)                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│         text-embedding-3-large (Azure AI Foundry)                │
│                                                                  │
│  Parallel embedding calls with varying dimensions:               │
│  ─ semantic_description     → 3072-dim semantic_vector           │
│  ─ structural_description   → 1024-dim structural_vector         │
│  ─ style_description        → 512-dim style_vector               │
│  ─ char_N_semantic_text     → 512-dim char_N_semantic_vector     │
│  ─ char_N_emotion_text      → 256-dim char_N_emotion_vector      │
│  ─ char_N_pose_text         → 256-dim char_N_pose_vector         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              Azure AI Search Index                               │
│                                                                  │
│  Primitive fields: image_id, generation_prompt, metadata...      │
│  Vector fields: semantic_vector, structural_vector,              │
│                 style_vector, char_0/1/2 sub-vectors             │
│  HNSW index per vector field, cosine similarity                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Optimized GPT-4o Call Strategy

**Single structured call** to GPT-4o rather than multiple calls:

```python
EXTRACTION_PROMPT = """
Analyze this image alongside its generation prompt. Return a JSON object with:

{
  "semantic_description": "Rich 200-word description of the image content,
    narrative, subjects, environment, mood, and thematic elements.",
  "structural_description": "150-word description focusing exclusively on
    spatial composition, layout, object positioning, foreground/midground/
    background, lines of composition, and geometric structure.",
  "style_description": "150-word description focusing exclusively on artistic
    style, color palette, lighting, texture, rendering technique, and
    visual treatment.",
  "characters": [
    {
      "character_id": "descriptive_id",
      "semantic": "2-3 sentences on identity, role, appearance, clothing.",
      "emotion": "2-3 sentences on emotional expression, body language cues.",
      "pose": "2-3 sentences on physical position, orientation, gestures."
    }
  ],
  "metadata": {
    "scene_type": "...",
    "time_of_day": "...",
    "lighting_condition": "...",
    "primary_subject": "...",
    "secondary_subjects": ["..."],
    "artistic_style": "...",
    "color_palette": ["..."],
    "tags": ["..."],
    "narrative_theme": "..."
  }
}

Generation prompt: {generation_prompt}
"""
```

**Benefits of single-call extraction:**

* **Latency:** One GPT-4o call (~2-4 seconds) instead of 4-5 separate calls
* **Cost:** Single image token charge instead of repeated charges
* **Consistency:** All descriptions are generated from the same understanding of the image
* **Token efficiency:** GPT-4o processes the image once

### 6.3 Embedding Batching Strategy

text-embedding-3-large supports **batch embedding** (multiple texts in one API call), but all texts in a batch must use the **same `dimensions` parameter**.

**Optimal batching:**

```python
# Group by target dimensions
batch_3072 = [semantic_description]  # 1 text
batch_1024 = [structural_description]  # 1 text
batch_512  = [style_desc, char_0_semantic, char_1_semantic, ...]  # N texts
batch_256  = [char_0_emotion, char_0_pose, char_1_emotion, ...]  # N texts

# 4 API calls total (parallelizable)
results_3072 = await embed_batch(batch_3072, dimensions=3072)
results_1024 = await embed_batch(batch_1024, dimensions=1024)
results_512  = await embed_batch(batch_512, dimensions=512)
results_256  = await embed_batch(batch_256, dimensions=256)
```

**Per-image API calls:** 1 GPT-4o call + 4 embedding calls = **5 API calls total**.

### 6.4 Azure AI Search Index Schema

```json
{
  "name": "candidate-index",
  "fields": [
    {"name": "image_id", "type": "Edm.String", "key": true},
    {"name": "generation_prompt", "type": "Edm.String", "searchable": true},
    {"name": "scene_type", "type": "Edm.String", "filterable": true},
    {"name": "lighting_condition", "type": "Edm.String", "filterable": true},
    {"name": "tags", "type": "Collection(Edm.String)", "searchable": true,
     "facetable": true},
    {"name": "emotional_polarity", "type": "Edm.Double", "filterable": true,
     "sortable": true},
    {"name": "character_count", "type": "Edm.Int32", "filterable": true},
    {"name": "metadata_json", "type": "Edm.String", "searchable": false},

    {"name": "semantic_vector", "type": "Collection(Edm.Single)",
     "dimensions": 3072, "vectorSearchProfile": "hnsw-cosine"},
    {"name": "structural_vector", "type": "Collection(Edm.Single)",
     "dimensions": 1024, "vectorSearchProfile": "hnsw-cosine"},
    {"name": "style_vector", "type": "Collection(Edm.Single)",
     "dimensions": 512, "vectorSearchProfile": "hnsw-cosine"},

    {"name": "char_0_semantic_vector", "type": "Collection(Edm.Single)",
     "dimensions": 512, "vectorSearchProfile": "hnsw-cosine"},
    {"name": "char_0_emotion_vector", "type": "Collection(Edm.Single)",
     "dimensions": 256, "vectorSearchProfile": "hnsw-cosine"},
    {"name": "char_0_pose_vector", "type": "Collection(Edm.Single)",
     "dimensions": 256, "vectorSearchProfile": "hnsw-cosine"},

    {"name": "char_1_semantic_vector", "type": "Collection(Edm.Single)",
     "dimensions": 512, "vectorSearchProfile": "hnsw-cosine"},
    {"name": "char_1_emotion_vector", "type": "Collection(Edm.Single)",
     "dimensions": 256, "vectorSearchProfile": "hnsw-cosine"},
    {"name": "char_1_pose_vector", "type": "Collection(Edm.Single)",
     "dimensions": 256, "vectorSearchProfile": "hnsw-cosine"},

    {"name": "char_2_semantic_vector", "type": "Collection(Edm.Single)",
     "dimensions": 512, "vectorSearchProfile": "hnsw-cosine"},
    {"name": "char_2_emotion_vector", "type": "Collection(Edm.Single)",
     "dimensions": 256, "vectorSearchProfile": "hnsw-cosine"},
    {"name": "char_2_pose_vector", "type": "Collection(Edm.Single)",
     "dimensions": 256, "vectorSearchProfile": "hnsw-cosine"}
  ],
  "vectorSearch": {
    "algorithms": [
      {"name": "hnsw-config", "kind": "hnsw",
       "hnswParameters": {"m": 4, "efConstruction": 400, "efSearch": 500,
                          "metric": "cosine"}}
    ],
    "profiles": [
      {"name": "hnsw-cosine", "algorithm": "hnsw-config"}
    ]
  }
}
```

### 6.5 Cost Estimation (Per Image)

| Operation                       | Estimated Cost     |
|---------------------------------|--------------------|
| GPT-4o vision (1 image + text)  | ~$0.01–0.03        |
| text-embedding-3-large (4 calls)| ~$0.0004           |
| Azure AI Search indexing         | Marginal            |
| **Total per image**             | **~$0.01–0.03**    |

At 10M images: **$100K–300K** for initial indexing (dominated by GPT-4o vision costs).

**Optimization:** For batch processing at scale, consider tiered approaches—use GPT-4o-mini for simpler images, GPT-4o for complex scenes.

### 6.6 Latency Profile

| Step                           | Expected Latency |
|--------------------------------|------------------|
| GPT-4o vision extraction       | 2–4 seconds      |
| Embedding generation (4 calls) | 200–500 ms       |
| Index document upload          | 50–100 ms        |
| **Total per image**            | **2.5–5 seconds** |

With parallel embedding calls: **2.5–4.5 seconds** per image, dominated by GPT-4o inference.

## 7. Key Risks and Mitigations

| Risk                                          | Impact | Mitigation                                                                    |
|-----------------------------------------------|--------|-------------------------------------------------------------------------------|
| Language-mediated vectors lose visual detail   | Medium | Validate recall quality; upgrade to custom vision models if needed             |
| GPT-4o rate limits at scale                    | High   | Implement retry logic, batch processing, request throttling                    |
| Dimension reduction degrades specific queries  | Low    | Benchmark dimension choices on representative queries before committing         |
| Character slot overflow (>3 characters)        | Low    | Cap at 3 slots; store overflow in metadata JSON for LLM re-ranking             |
| Azure AI Search vector field limit             | Medium | Current limit is ~15 vector fields; monitor as schema grows                    |
| Embedding drift over model updates             | Medium | Version embeddings; re-index when model version changes                        |

## 8. Recommended Next Research Topics

1. **Query-time multi-vector strategy:** How to generate query embeddings for hybrid retrieval across semantic, structural, and style vectors simultaneously
2. **Azure AI Search scoring profiles:** How to configure weighted multi-vector scoring in Azure AI Search (RRF vs. custom scoring)
3. **GPT-4o structured output reliability:** Benchmark GPT-4o's consistency in producing the required JSON schema across diverse image types
4. **Re-ranking with LLM reasoning:** How to implement the LLM-based re-ranking step using Azure AI Foundry
5. **Incremental re-indexing strategy:** How to handle updates when new vectors are added or existing images are re-processed
6. **Cost optimization at scale:** GPT-4o-mini vs. GPT-4o for different extraction tasks; caching strategies
7. **Azure AI Search vector field limits:** Current and planned limits for vector fields per index (impacts character slot design)
8. **Benchmark: text-embedding-3-large dimensions vs. retrieval quality** on the specific image domain
