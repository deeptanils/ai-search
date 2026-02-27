# Candidate Generation & AI Search Pipeline - Requirements Document

## 1. Objective

Design and implement an end-to-end pipeline for:

1.  Candidate Generation
2.  AI-powered Search using Azure AI Search

The system must: - Accept an image and its generation prompt as input. -
Generate synthetic metadata using an LLM. - Extract structured
multi-dimensional representations from image + prompt. - Encode
multi-vector embeddings. - Index everything into Azure AI Search for
hybrid and vector retrieval.

------------------------------------------------------------------------

## 2. High-Level Architecture

Input: - Image - LLM prompt used to generate the image

Pipeline Stages: 1. Synthetic Metadata Generation (LLM-based) 2. Image +
Prompt Understanding Layer 3. Semantic & Narrative Extraction 4.
Multi-Vector Encoding Layer 5. Azure AI Search Indexing 6. Candidate
Retrieval Layer

------------------------------------------------------------------------

## 3. Input Specification

### 3.1 Raw Inputs

-   image_id: string
-   image_binary / image_url
-   generation_prompt: string

### 3.2 Synthetic Metadata (Generated via LLM)

For each image, generate structured synthetic metadata including:

-   scene_type
-   time_of_day
-   lighting_condition
-   camera_angle
-   environment
-   primary_subject
-   secondary_subjects
-   style_descriptor
-   artistic_influence
-   color_palette
-   tags (list of keywords)

Output format (JSON):

{ "image_id": "...", "scene_type": "...", "time_of_day": "...",
"lighting_condition": "...", "primary_subject": "...",
"style_descriptor": "...", "tags": \[...\] }

------------------------------------------------------------------------

## 4. Image + Prompt Understanding Layer

Extract information from:

1.  The image
2.  The original generation prompt

### 4.1 Extracted Dimensions

#### 4.1.1 Narrative Intent

-   What story is being told?
-   Is it cinematic, documentary, surreal, fantasy, romantic, etc.?

#### 4.1.2 Character States

-   Physical state (standing, running, injured, relaxed)
-   Social state (alone, interacting, confrontational)
-   Cognitive state (thinking, observing, unaware)

#### 4.1.3 Emotional Trajectory

-   Starting emotion
-   Mid-scene emotional shift
-   End emotional tone
-   Emotional polarity score

#### 4.1.4 Required Objects

-   Key objects central to the scene
-   Supporting contextual objects
-   Background symbolic elements

#### 4.1.5 Low-Light Robustness Indicators

-   Brightness score
-   Contrast score
-   Noise estimate
-   Shadow dominance
-   Visibility confidence score

#### 4.1.6 Character / Object Details

-   Clothing attributes
-   Accessories
-   Pose and orientation
-   Facial attributes
-   Texture and material descriptors

All extracted outputs must be structured JSON.

------------------------------------------------------------------------

## 5. Multi-Vector Encoding Layer

The system must generate multiple embedding vectors per image.

### 5.1 Semantic Vector

Model: SigLIP / EVA-CLIP Purpose: - Capture semantic alignment between
image and text. - Used for primary retrieval.

Field: semantic_vector (float array)

------------------------------------------------------------------------

### 5.2 Structural Vector

Model: DINOv2 Purpose: - Capture layout, spatial composition, object
positioning. - Useful for structural similarity search.

Field: structural_vector (float array)

------------------------------------------------------------------------

### 5.3 Style Vector

Model: Style Encoder / LoRA-derived embedding Purpose: - Capture
artistic style, lighting tone, color treatment.

Field: style_vector (float array)

------------------------------------------------------------------------

### 5.4 Character Sub-Vectors

For each detected primary character:

-   character_semantic_vector
-   character_emotion_vector
-   character_pose_vector

Store as nested array:

"character_vectors": \[ { "character_id": "...", "semantic": \[...\],
"emotion": \[...\], "pose": \[...\] }\]

------------------------------------------------------------------------

## 6. Azure AI Search Index Design

### 6.1 Index Fields

Primitive Fields: - image_id (key) - generation_prompt (searchable) -
scene_type (filterable) - lighting_condition (filterable) - tags
(searchable, facetable) - emotional_polarity (filterable, sortable) -
low_light_score (filterable)

Vector Fields: - semantic_vector (vector search enabled) -
structural_vector (vector search enabled) - style_vector (vector search
enabled) - character_vectors.semantic (vector search enabled)

All vectors must use Azure AI Search vector configuration with: - Cosine
similarity - HNSW indexing

------------------------------------------------------------------------

## 7. Retrieval Strategy

### 7.1 Hybrid Search

Combine: - Keyword search (BM25) - Semantic vector search - Structural
vector search - Style vector search

Final score = weighted combination:

FinalScore = w1 \* semantic_similarity + w2 \* structural_similarity +
w3 \* style_similarity + w4 \* keyword_score

Weights must be configurable.

------------------------------------------------------------------------

### 7.2 Candidate Generation Flow

1.  User query (text or image).
2.  Generate query embedding(s).
3.  Perform parallel vector searches.
4.  Merge candidates.
5.  Re-rank using:
    -   Emotional alignment
    -   Narrative consistency
    -   Object overlap
    -   Low-light compatibility

------------------------------------------------------------------------

## 8. Scalability & Performance Requirements

-   Support \>= 10M indexed images.
-   P95 query latency \< 300 ms.
-   Support batch indexing.
-   Embeddings stored in compressed float format if needed.
-   Enable incremental re-indexing.

------------------------------------------------------------------------

## 9. Observability

Track: - Embedding generation latency - Indexing latency - Retrieval
latency - Candidate diversity metrics - Drift in embedding distributions

------------------------------------------------------------------------

## 10. Future Enhancements

-   Temporal stability scoring (for video extension)
-   Identity drift risk scoring
-   Texture frequency analysis module
-   Cross-modal reasoning agent for fraud/misuse detection
-   Online re-ranking with LLM reasoning

------------------------------------------------------------------------

## 11. Deliverables

1.  Data schema definition
2.  Embedding generation service
3.  Azure AI Search index configuration
4.  Retrieval service API
5.  Evaluation framework for retrieval quality





# Candidate Generation & AI Search Pipeline - Requirements Document (v2)

---

# 1. Objective

Design and implement a production-grade pipeline for:

1. Candidate Generation
2. AI-powered Search using Microsoft Azure AI Search

This system must:

- Accept image + generation prompt as input
- Generate synthetic metadata via LLM (Azure AI Foundry)
- Extract structured narrative + emotional + structural information
- Generate multi-vector embeddings
- Index into Azure AI Search
- Support hybrid retrieval and candidate re-ranking

All LLMs and embedding models MUST be served through Azure AI Foundry.

---

# 2. Technology Stack

## 2.1 Model Hosting

ALL LLM and embedding models must be accessed via Azure AI Foundry.

Example Endpoint:
https://can-foundry.cognitiveservices.azure.com/

Models:
- LLM (for metadata + extraction)
- text-embedding-3-large (semantic embeddings)
- Future multimodal embeddings (CLIP/SigLIP variants)

---

## 2.2 Vector Database & Search

Vector DB: Microsoft Azure AI Search

Required capabilities:
- HNSW vector index
- Cosine similarity
- Hybrid search (BM25 + Vector)
- Multiple vector fields per document
- Filterable structured metadata
- Scoring profiles
- Faceting
- Re-ranking

---

# 3. Configuration & Secrets Management

## 3.1 .env File (Secrets Only)

All API keys must be stored in a `.env` file.

Example:

AZURE_FOUNDRY_ENDPOINT=https://can-foundry.cognitiveservices.azure.com/
AZURE_FOUNDRY_API_KEY=<YOUR_API_KEY>
AZURE_OPENAI_API_VERSION=2024-12-01-preview

AZURE_AI_SEARCH_ENDPOINT=https://<search-service>.search.windows.net
AZURE_AI_SEARCH_API_KEY=<SEARCH_API_KEY>
AZURE_AI_SEARCH_INDEX_NAME=candidate-index

No secrets must exist inside code.

---

## 3.2 config.yaml (Non-secret Configuration)

All configuration must be stored in config.yaml.

Example:

models:
  embedding_model: text-embedding-3-large
  llm_model: gpt-4o

search:
  semantic_weight: 0.5
  structural_weight: 0.2
  style_weight: 0.2
  keyword_weight: 0.1

index:
  vector_dimensions:
    semantic: 3072
    structural: 1024
    style: 512

Before running test cases:
- Populate `.env`
- Update config.yaml placeholders

---

# 4. Input Specification

- image_id
- image_url or image_binary
- generation_prompt

---

# 5. Synthetic Metadata Generation (LLM via Azure Foundry)

Generate structured JSON metadata:

- scene_type
- time_of_day
- lighting_condition
- primary_subject
- secondary_subjects
- artistic_style
- color_palette
- tags
- narrative_theme

---

# 6. Extraction Layer

Extract:

## Narrative Intent
## Character States
## Emotional Trajectory
## Required Objects
## Low-Light Robustness Metrics
## Character/Object Attributes

All outputs must be structured JSON.

---

# 7. Multi-Vector Encoding Layer

All embeddings must be generated via Azure Foundry.

Required vectors:

- semantic_vector
- structural_vector
- style_vector
- character_sub_vectors

---

# 8. Azure AI Search Index Design

Primitive Fields:
- image_id (key)
- generation_prompt (searchable)
- scene_type (filterable)
- lighting_condition (filterable)
- tags (facetable)
- emotional_polarity (sortable)

Vector Fields:
- semantic_vector
- structural_vector
- style_vector
- character_vectors.semantic

Vector config:
- HNSW
- Cosine similarity

---

# 9. Retrieval Flow

1. Generate query embeddings via Azure Foundry
2. Perform parallel vector searches
3. Hybrid merge
4. Weighted scoring
5. Optional LLM re-rank

Weights configurable in config.yaml.

---

# 10. Python Project Management

Language: Python
Dependency Manager: UV

---

# 11. Repository Structure

project/
├── src/
│   ├── ingestion/
│   ├── extraction/
│   ├── embeddings/
│   ├── indexing/
│   ├── retrieval/
├── config.yaml
├── .env
├── pyproject.toml
└── tests/

---

# 12. Mandatory Rules

- No hardcoded secrets
- All config externalized
- Azure Foundry only for models
- Azure AI Search only for vector search
- Index schema versioned

---

# 13. Future Extensions

- Identity drift risk
- Texture frequency analysis
- Temporal stability
- Multi-modal re-ranking
