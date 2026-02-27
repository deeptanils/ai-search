---
title: Azure AI Foundry SDK Research for Python
description: Comprehensive research on Azure AI Foundry Python SDK capabilities for LLM calls, embeddings, vision, rate limiting, and package selection
author: copilot-research-subagent
ms.date: 2026-02-26
ms.topic: reference
keywords:
  - azure ai foundry
  - openai sdk
  - embeddings
  - gpt-4o
  - text-embedding-3-large
  - azure ai search
  - python
estimated_reading_time: 12
---

## Executive Summary

This document captures research findings on Azure AI Foundry (formerly Azure OpenAI Service) Python SDK capabilities relevant to the Candidate Generation and AI Search Pipeline. The pipeline requires LLM-based metadata generation, multimodal image analysis, embedding generation, and batch processing at scale (10M+ images). All models must be served through Azure AI Foundry per the project requirements.

The recommended approach uses the `openai` Python SDK with the `AzureOpenAI` client class, targeting API version `2024-12-01-preview`. This provides unified access to GPT-4o (chat completions with vision and structured JSON output) and `text-embedding-3-large` (with configurable dimensions).

## 1. Python SDK Package Options

Three primary SDK options exist for interacting with Azure AI Foundry. Each serves different use cases.

### 1.1 OpenAI Python SDK (Recommended)

* Package: `openai>=1.58.0`
* Provides `AzureOpenAI` and `AsyncAzureOpenAI` client classes
* Supports all OpenAI models deployed on Azure AI Foundry: GPT-4o, text-embedding-3-large, DALL-E, Whisper
* Feature parity with OpenAI's API surface including Structured Outputs, vision, and batch API
* Most mature, best documented, widest community adoption
* Async support via `AsyncAzureOpenAI` for concurrent processing

```python
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint="https://can-foundry.cognitiveservices.azure.com/",
    api_key="<AZURE_FOUNDRY_API_KEY>",
    api_version="2024-12-01-preview"
)
```

### 1.2 Azure AI Inference SDK

* Package: `azure-ai-inference>=1.0.0b7`
* Model-agnostic SDK for Azure AI Foundry and Azure AI Model Catalog
* Uses `ChatCompletionsClient` and `EmbeddingsClient`
* Designed for non-OpenAI models (Mistral, Llama, Cohere, Phi) hosted on Azure AI Foundry
* Less mature for OpenAI models specifically; the `openai` SDK remains the preferred path for GPT-4o and text-embedding-3-large

```python
from azure.ai.inference import ChatCompletionsClient, EmbeddingsClient
from azure.core.credentials import AzureKeyCredential

chat_client = ChatCompletionsClient(
    endpoint="https://can-foundry.cognitiveservices.azure.com/",
    credential=AzureKeyCredential("<API_KEY>")
)

embed_client = EmbeddingsClient(
    endpoint="https://can-foundry.cognitiveservices.azure.com/",
    credential=AzureKeyCredential("<API_KEY>")
)
```

### 1.3 Azure Identity SDK

* Package: `azure-identity>=1.17.0`
* Provides token-based authentication via Microsoft Entra ID (formerly Azure AD)
* Useful for production deployments where API key rotation is undesirable
* Compatible with both `openai` and `azure-ai-inference` SDKs

```python
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    azure_endpoint="https://can-foundry.cognitiveservices.azure.com/",
    azure_ad_token_provider=token_provider,
    api_version="2024-12-01-preview"
)
```

### 1.4 Recommendation for This Project

Use `openai` as the primary SDK for all LLM and embedding calls. Add `azure-identity` for production deployments needing Entra ID authentication. The `azure-ai-inference` SDK is unnecessary unless the project adopts non-OpenAI models from the Azure AI Model Catalog in the future.

Required `pyproject.toml` dependencies:

```toml
[project]
dependencies = [
    "openai>=1.58.0",
    "azure-identity>=1.17.0",
    "python-dotenv>=1.0.0",
]
```

## 2. LLM Calls via Azure AI Foundry

### 2.1 Initialization Pattern

```python
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_FOUNDRY_ENDPOINT"),
    api_key=os.getenv("AZURE_FOUNDRY_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
)
```

### 2.2 Chat Completions with JSON Output

Two approaches exist for structured JSON generation. Structured Outputs with JSON Schema is the preferred option for the metadata generation use case because it guarantees schema conformance.

#### Option A: JSON Mode (Basic)

Forces the model to produce valid JSON but does not enforce a specific schema.

```python
response = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},
    messages=[
        {
            "role": "system",
            "content": (
                "You are a metadata extraction assistant. "
                "Always respond with valid JSON matching the requested schema."
            )
        },
        {
            "role": "user",
            "content": "Generate metadata for this scene: A lone astronaut on Mars at sunset."
        }
    ],
    temperature=0.2,
    max_tokens=1000
)

import json
metadata = json.loads(response.choices[0].message.content)
```

#### Option B: Structured Outputs with JSON Schema (Preferred)

Guarantees the output conforms to a provided JSON Schema. Available in API version `2024-08-01-preview` and later.

```python
from pydantic import BaseModel
from typing import Optional

class ImageMetadata(BaseModel):
    scene_type: str
    time_of_day: str
    lighting_condition: str
    primary_subject: str
    secondary_subjects: list[str]
    artistic_style: str
    color_palette: list[str]
    tags: list[str]
    narrative_theme: str

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": "Extract structured metadata from the provided image description."
        },
        {
            "role": "user",
            "content": "A lone astronaut on Mars at sunset, cinematic framing."
        }
    ],
    response_format=ImageMetadata,
    temperature=0.2
)

metadata: ImageMetadata = response.choices[0].message.parsed
```

> [!IMPORTANT]
> The `client.beta.chat.completions.parse()` method with Pydantic models requires `openai>=1.50.0`. The SDK auto-generates the JSON Schema from the Pydantic model and passes it to the API.

### 2.3 Key Parameters for Metadata Generation

| Parameter          | Value             | Rationale                                                     |
|--------------------|-------------------|---------------------------------------------------------------|
| `model`            | `gpt-4o`          | Required model per project config                             |
| `temperature`      | `0.1` to `0.3`    | Low temperature for consistent, deterministic metadata output |
| `max_tokens`       | `1000` to `2000`  | Sufficient for structured metadata payloads                   |
| `response_format`  | JSON Schema or JSON Mode | Enforces valid JSON output                             |
| `seed`             | Fixed integer      | Improves reproducibility across runs                          |

## 3. Embedding Generation

### 3.1 Basic Embedding Call

```python
response = client.embeddings.create(
    model="text-embedding-3-large",
    input="A cinematic scene of an astronaut on Mars at sunset",
    dimensions=3072  # Full dimensionality
)

embedding = response.data[0].embedding  # list[float] of length 3072
```

### 3.2 Dimension Options

The `text-embedding-3-large` model supports a `dimensions` parameter that truncates the embedding via Matryoshka Representation Learning. This allows trading dimensionality for storage and query speed without retraining.

| Purpose               | Dimensions | Field in Index       | Notes                              |
|-----------------------|------------|----------------------|------------------------------------|
| Semantic vector       | 3072       | `semantic_vector`    | Full dimensionality, max accuracy  |
| Structural vector     | 1024       | `structural_vector`  | Reduced for layout/spatial signals |
| Style vector          | 512        | `style_vector`       | Compact for style matching         |
| Character sub-vectors | 1024       | `character_vectors`  | Per-character embeddings           |

> [!NOTE]
> The `dimensions` parameter is available only for `text-embedding-3-large` and `text-embedding-3-small`. Older `text-embedding-ada-002` does not support this parameter.

### 3.3 Batch Embedding

The embeddings API accepts a list of inputs, processing up to 2048 strings in a single request.

```python
texts = [
    "Scene: astronaut on Mars, cinematic sunset",
    "Style: warm amber tones, lens flare, wide angle",
    "Character: solitary figure in spacesuit, contemplative pose"
]

response = client.embeddings.create(
    model="text-embedding-3-large",
    input=texts,
    dimensions=3072
)

embeddings = [item.embedding for item in response.data]
# embeddings[0] corresponds to texts[0], etc.
```

### 3.4 Batch Processing Considerations

* Maximum input list size: 2048 strings per request
* Maximum total tokens per request: 8191 tokens per input string
* Each input string should stay under 8191 tokens; longer strings are truncated
* For batch efficiency, group inputs into chunks of 1000 to 2048 strings

```python
import asyncio
from openai import AsyncAzureOpenAI

async_client = AsyncAzureOpenAI(
    azure_endpoint=os.getenv("AZURE_FOUNDRY_ENDPOINT"),
    api_key=os.getenv("AZURE_FOUNDRY_API_KEY"),
    api_version="2024-12-01-preview"
)

async def embed_batch(texts: list[str], dimensions: int = 3072) -> list[list[float]]:
    """Embed a batch of texts with chunking for API limits."""
    chunk_size = 2048
    all_embeddings = []
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        response = await async_client.embeddings.create(
            model="text-embedding-3-large",
            input=chunk,
            dimensions=dimensions
        )
        all_embeddings.extend([item.embedding for item in response.data])
    return all_embeddings
```

## 4. Multimodal and Vision Capabilities

GPT-4o natively supports image inputs in chat completions. This is critical for the extraction layer (narrative intent, character states, emotional trajectory, required objects, low-light metrics, and character/object attributes).

### 4.1 Image Input via URL

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Analyze this image and return structured JSON with: "
                        "narrative_intent, character_states, emotional_trajectory, "
                        "required_objects, low_light_metrics, character_attributes."
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://storage.example.com/images/img_001.jpg",
                        "detail": "high"
                    }
                }
            ]
        }
    ],
    response_format={"type": "json_object"},
    max_tokens=4096,
    temperature=0.2
)
```

### 4.2 Image Input via Base64

```python
import base64

def encode_image_base64(image_path: str) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")

image_b64 = encode_image_base64("/path/to/image.jpg")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Extract narrative intent and emotional trajectory from this image."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}",
                        "detail": "high"
                    }
                }
            ]
        }
    ],
    response_format={"type": "json_object"},
    max_tokens=4096
)
```

### 4.3 Image Detail Levels

| Detail Level | Token Cost     | Resolution              | Use Case                                    |
|--------------|----------------|-------------------------|---------------------------------------------|
| `low`        | 85 tokens      | 512x512 fixed           | Quick classification, basic scene detection  |
| `high`       | 85 + 170/tile  | Up to 2048x2048, tiled  | Detailed analysis, character attributes      |
| `auto`       | Varies         | Model decides           | General purpose                              |

For the extraction layer requiring character attributes, facial details, and low-light metrics, `high` detail is necessary. For batch metadata generation where scene-level features suffice, `low` detail reduces cost by approximately 10x.

### 4.4 Combined Image and Prompt Analysis

The pipeline requires analyzing both the image and its generation prompt together. Send both in a single message:

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": (
                "You are an image analysis system. Given an image and the prompt "
                "that generated it, extract structured metadata. Return JSON."
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Generation prompt: {generation_prompt}\n\nAnalyze the image and prompt together."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                        "detail": "high"
                    }
                }
            ]
        }
    ],
    response_format={"type": "json_object"},
    max_tokens=4096,
    temperature=0.2
)
```

### 4.5 Vision Limitations

* Maximum image size: 20 MB per image
* Supported formats: JPEG, PNG, GIF (first frame only), WebP
* Maximum images per request: 10 (varies by deployment configuration)
* Token counting for images: low detail = 85 tokens; high detail = 85 + 170 per 512x512 tile
* The model cannot identify specific real people. Character analysis is limited to posture, clothing, expressions, and spatial relationships.

## 5. API Version and SDK Compatibility

### 5.1 API Version `2024-12-01-preview`

The project specifies `AZURE_OPENAI_API_VERSION=2024-12-01-preview`. This version supports:

| Feature                          | Supported | Notes                                           |
|----------------------------------|-----------|-------------------------------------------------|
| GPT-4o chat completions          | Yes       | Full support including streaming                 |
| Vision/multimodal inputs         | Yes       | Image URLs and base64                            |
| JSON mode (`json_object`)        | Yes       | Basic structured output                          |
| Structured Outputs (JSON Schema) | Yes       | Schema-enforced output via `response_format`     |
| `text-embedding-3-large`         | Yes       | With `dimensions` parameter                      |
| Batch API                        | Yes       | Async batch processing for large-scale workloads |
| Assistants API                   | Yes       | Not needed for this project                      |
| Function calling                 | Yes       | Alternative to structured outputs                |

### 5.2 SDK Version Requirements

| Package            | Minimum Version | Required For                                    |
|--------------------|-----------------|-------------------------------------------------|
| `openai`           | `>=1.58.0`      | Structured Outputs, vision, batch API, bug fixes |
| `azure-identity`   | `>=1.17.0`      | Entra ID token provider                          |
| `python-dotenv`    | `>=1.0.0`       | Loading `.env` configuration                     |
| `pydantic`         | `>=2.0.0`       | Structured Output schema definitions             |
| `httpx`            | `>=0.25.0`      | Async HTTP client (openai dependency)            |

### 5.3 Model Deployment Names

When deploying models in Azure AI Foundry, the `model` parameter in API calls refers to the deployment name, not the model name. Ensure config.yaml deployment names match:

```yaml
models:
  llm_model: gpt-4o          # This must match the deployment name in Azure AI Foundry
  embedding_model: text-embedding-3-large  # Same: must match deployment name
```

## 6. Rate Limiting and Batch Processing for 10M+ Images

### 6.1 Azure OpenAI Rate Limits

Rate limits are expressed in two dimensions:

| Dimension               | Typical Limit (S0 tier)    | Notes                           |
|-------------------------|----------------------------|---------------------------------|
| Requests per minute     | 60 to 1000 RPM            | Varies by model and region      |
| Tokens per minute       | 80K to 450K TPM           | Combined input + output tokens  |

Headers returned with each response:

* `x-ratelimit-remaining-requests`
* `x-ratelimit-remaining-tokens`
* `retry-after` (seconds, when rate-limited)

### 6.2 Provisioned Throughput Units (PTU)

For 10M+ image pipelines, standard pay-per-token pricing with shared rate limits is insufficient. Provisioned Throughput Units provide:

* Guaranteed, dedicated capacity
* Predictable latency
* No rate limiting within provisioned capacity
* Cost optimization at scale (pre-purchase compute)

Recommendation: deploy GPT-4o and text-embedding-3-large with PTU allocation for the batch indexing pipeline.

### 6.3 Azure OpenAI Batch API

The Batch API is designed for large-scale, non-real-time processing. It accepts JSONL files of requests and processes them asynchronously with a 24-hour SLA.

Benefits:

* 50% cost reduction compared to standard API pricing
* Higher rate limits (separate from synchronous limits)
* Automatic retries and error handling
* Scales to millions of requests

```python
# 1. Prepare batch input file (JSONL format)
import json

batch_requests = []
for i, image_url in enumerate(image_urls):
    batch_requests.append({
        "custom_id": f"image-{i}",
        "method": "POST",
        "url": "/chat/completions",
        "body": {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract metadata as JSON."},
                        {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}}
                    ]
                }
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 2000
        }
    })

# Write JSONL file
with open("batch_input.jsonl", "w") as f:
    for req in batch_requests:
        f.write(json.dumps(req) + "\n")

# 2. Upload and submit batch
batch_input_file = client.files.create(
    file=open("batch_input.jsonl", "rb"),
    purpose="batch"
)

batch = client.batches.create(
    input_file_id=batch_input_file.id,
    endpoint="/chat/completions",
    completion_window="24h"
)

# 3. Poll for completion
import time
while True:
    status = client.batches.retrieve(batch.id)
    if status.status in ("completed", "failed", "expired"):
        break
    time.sleep(60)

# 4. Download results
if status.status == "completed":
    result_file = client.files.content(status.output_file_id)
    results = [json.loads(line) for line in result_file.text.strip().split("\n")]
```

> [!IMPORTANT]
> The Batch API supports vision inputs (image URLs) in batch requests. For base64-encoded images, upload them to Azure Blob Storage first and use SAS URLs, since embedding large base64 strings in JSONL files creates impractically large batch files.

### 6.4 Async Concurrency with Rate Limit Handling

For real-time or near-real-time processing (not batch API), use async concurrency with semaphore-based throttling and exponential backoff.

```python
import asyncio
import random
from openai import AsyncAzureOpenAI, RateLimitError

async_client = AsyncAzureOpenAI(
    azure_endpoint=os.getenv("AZURE_FOUNDRY_ENDPOINT"),
    api_key=os.getenv("AZURE_FOUNDRY_API_KEY"),
    api_version="2024-12-01-preview"
)

# Semaphore limits concurrent requests
semaphore = asyncio.Semaphore(50)

async def call_with_backoff(func, max_retries: int = 5):
    """Execute an async function with exponential backoff on rate limits."""
    for attempt in range(max_retries):
        try:
            async with semaphore:
                return await func()
        except RateLimitError as e:
            retry_after = float(e.response.headers.get("retry-after", 1))
            wait = retry_after + random.uniform(0, 1)
            await asyncio.sleep(wait)
    raise Exception("Max retries exceeded")

async def process_image(image_url: str, prompt: str) -> dict:
    """Process a single image through the extraction pipeline."""
    async def make_call():
        return await async_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Analyze: {prompt}"},
                        {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}}
                    ]
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=4096
        )
    return await call_with_backoff(make_call)
```

### 6.5 Processing Strategy for 10M+ Images

A tiered approach optimizes cost and throughput:

| Stage                     | Method                    | Concurrency   | Cost Impact      |
|---------------------------|---------------------------|---------------|------------------|
| Metadata generation (LLM) | Batch API                 | Managed by API | 50% savings      |
| Image extraction (Vision)  | Batch API + async fallback | Managed/50 concurrent | 50% savings |
| Embedding generation       | Async with chunking       | 100 concurrent | Standard pricing |
| Index upload               | Azure SDK bulk push       | 1000 docs/batch | N/A             |

Estimated throughput (with PTU deployment):

* Embeddings: approximately 500 to 1000 requests/second (batched)
* LLM metadata: approximately 100 to 300 requests/second (batch API)
* Vision extraction: approximately 50 to 100 requests/second (batch API)

For 10M images at 100 requests/second (vision), the vision extraction stage alone takes approximately 28 hours. The Batch API's 24-hour window is ideal for this scale.

## 7. Code Architecture Recommendations

Based on this research, the recommended client initialization pattern:

```python
# src/clients/azure_foundry.py
import os
from functools import lru_cache
from openai import AzureOpenAI, AsyncAzureOpenAI
from dotenv import load_dotenv

load_dotenv()

@lru_cache(maxsize=1)
def get_sync_client() -> AzureOpenAI:
    """Return a cached synchronous Azure OpenAI client."""
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_FOUNDRY_ENDPOINT"],
        api_key=os.environ["AZURE_FOUNDRY_API_KEY"],
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    )

@lru_cache(maxsize=1)
def get_async_client() -> AsyncAzureOpenAI:
    """Return a cached asynchronous Azure OpenAI client."""
    return AsyncAzureOpenAI(
        azure_endpoint=os.environ["AZURE_FOUNDRY_ENDPOINT"],
        api_key=os.environ["AZURE_FOUNDRY_API_KEY"],
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    )
```

## 8. Open Questions and Recommended Next Research

### 8.1 Immediate Follow-Up Research

1. Azure AI Search Python SDK (`azure-search-documents`): Index creation, vector field configuration, HNSW parameters, hybrid search, scoring profiles, and bulk document upload patterns.
2. SigLIP/EVA-CLIP and DINOv2 integration: The requirements mention these models for semantic and structural vectors. Research whether Azure AI Foundry hosts these models or whether they require local inference (e.g., via `transformers` or `timm`).
3. Character sub-vector extraction: Research pose estimation models compatible with Azure AI Foundry or local inference for generating character_pose_vector.

### 8.2 Architecture Decisions Needed

* Whether to use the Batch API versus async concurrency for the initial 10M image ingestion
* PTU sizing: calculate required PTUs based on throughput needs and cost modeling
* Whether `text-embedding-3-large` at reduced dimensions (1024, 512) produces sufficient quality for structural and style vectors, or whether separate specialized models are needed
* Image storage strategy: Azure Blob Storage with SAS URLs for batch API image references

### 8.3 Cost Estimation Research

* GPT-4o pricing per token (input: images at low/high detail, text; output)
* text-embedding-3-large pricing per token
* Batch API 50% discount applicability
* PTU pricing versus pay-per-token breakeven analysis at 10M scale
