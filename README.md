# AI Search Pipeline

Candidate Generation & AI Search Pipeline using Azure AI Foundry, Azure AI Search, and Azure Blob Storage. Accepts images with generation prompts, extracts structured metadata via GPT-4o, generates multi-vector embeddings, uploads images to cloud storage, indexes into Azure AI Search, and supports hybrid multi-vector retrieval with relevance scoring.

## Prerequisites

* Python 3.11+
* [UV](https://docs.astral.sh/uv/) package manager
* [Azure CLI](https://aka.ms/install-azure-cli) (for resource provisioning)
* Azure subscription with the resources listed in the [Azure Resources](#azure-resources) section

### Azure Resources

The pipeline requires these Azure services:

| Resource | Service | SKU / Tier | Purpose |
|----------|---------|------------|---------|
| **Azure AI Foundry** | Cognitive Services (OpenAI) | S0 | Hosts model deployments for extraction, embeddings, and image generation |
| **Azure AI Search** | Search | Basic | Vector + keyword index with HNSW for hybrid retrieval |
| **Azure Blob Storage** | Storage Account (StorageV2) | Standard_LRS | Cloud storage for ingested images |
| **Azure Computer Vision** | Cognitive Services | S1 (optional) | Florence backend — only if not using Foundry for image embeddings |

#### Model Deployments

These models must be deployed inside the Azure AI Foundry resource:

| Deployment Name | Model | Version | Type | Purpose |
|-----------------|-------|---------|------|---------|
| `gpt-4o` | GPT-4o | `2024-11-20` | Global Standard | Image analysis and structured metadata extraction |
| `text-embedding-3-large` | text-embedding-3-large | `1` | Standard | Text embeddings (semantic 3072d, structural 1024d, style 512d) |
| `gpt-image-1.5` | gpt-image-1 | `2025-04-15` | Standard | Ramayana image generation (optional) |
| `embed-v-4-0` | Cohere Embed v4 | — | Serverless (MaaS) | Image embeddings (1024d) — deploy via AI Foundry portal |

> [!NOTE]
> Cohere Embed v4 (`embed-v-4-0`) is a **serverless** (Models-as-a-Service) deployment.
> It must be deployed manually through the [Azure AI Foundry portal](https://ai.azure.com)
> under **Model catalog → Cohere → embed-v-4-0 → Deploy**.

#### Automated Resource Provisioning

Use the included setup script to create all resources automatically:

```bash
chmod +x scripts/setup_azure.sh
./scripts/setup_azure.sh
```

The script creates the resource group, AI Foundry account with model deployments, AI Search service, Blob Storage account with container, assigns RBAC roles, and generates a `.env` file. Customise names and SKUs by exporting environment variables before running:

```bash
export RESOURCE_GROUP=my-rg
export LOCATION=westus2
export FOUNDRY_ACCOUNT_NAME=my-foundry
./scripts/setup_azure.sh
```

See [scripts/setup_azure.sh](scripts/setup_azure.sh) for the full list of configuration variables.

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd ai-search

# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install all dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

> [!IMPORTANT]
> Always use `source .venv/bin/activate` before running commands. Do not use `uv run` — the project requires the activated venv for correct module resolution.

### macOS SSL Certificate Fix

On macOS, the venv Python may fail to verify Azure HTTPS endpoints. Set this environment variable before any network operation:

```bash
export SSL_CERT_FILE=/private/etc/ssl/cert.pem
```

Add it to your shell profile (`.zshrc` or `.bashrc`) to persist across sessions.

## Configuration

### Environment Variables

Copy the example `.env` file and fill in your Azure credentials:

```bash
cp .env.example .env
```

Edit `.env` with your values:

```dotenv
# Azure AI Foundry (OpenAI + Inference)
AZURE_FOUNDRY_ENDPOINT=https://your-foundry.cognitiveservices.azure.com/
AZURE_FOUNDRY_EMBED_ENDPOINT=https://your-foundry.services.ai.azure.com/models
AZURE_FOUNDRY_API_KEY=your-api-key-here
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# Azure AI Search
AZURE_AI_SEARCH_ENDPOINT=https://your-search.search.windows.net
AZURE_AI_SEARCH_API_KEY=your-search-admin-key
AZURE_AI_SEARCH_INDEX_NAME=candidate-index

# Azure Blob Storage
AZURE_STORAGE_ACCOUNT_NAME=your-storage-account-name
AZURE_STORAGE_CONTAINER_NAME=images

# Azure Computer Vision (optional — only for Florence backend)
AZURE_CV_ENDPOINT=https://your-cv-resource.cognitiveservices.azure.com
AZURE_CV_API_KEY=your-cv-api-key-here
```

| Variable                         | Required | Purpose                                      |
|----------------------------------|----------|----------------------------------------------|
| `AZURE_FOUNDRY_ENDPOINT`         | **Yes**  | Azure AI Foundry base endpoint (OpenAI API)  |
| `AZURE_FOUNDRY_EMBED_ENDPOINT`   | **Yes**  | Foundry models endpoint for Inference SDK (Cohere Embed v4) |
| `AZURE_FOUNDRY_API_KEY`          | No       | API key fallback (Entra ID preferred)        |
| `AZURE_OPENAI_API_VERSION`       | No       | OpenAI API version (default: `2024-12-01-preview`) |
| `AZURE_AI_SEARCH_ENDPOINT`       | **Yes**  | Azure AI Search service endpoint             |
| `AZURE_AI_SEARCH_API_KEY`        | **Yes**  | Azure AI Search admin API key                |
| `AZURE_AI_SEARCH_INDEX_NAME`     | No       | Index name (default: `candidate-index`)      |
| `AZURE_STORAGE_ACCOUNT_NAME`     | No       | Azure Blob Storage account name (enables cloud image storage) |
| `AZURE_STORAGE_CONTAINER_NAME`   | No       | Blob container name (default: `images`)      |
| `AZURE_CV_ENDPOINT`              | No       | Computer Vision endpoint (Florence only)     |
| `AZURE_CV_API_KEY`               | No       | Computer Vision API key (Florence only)      |

### Authentication

The pipeline uses **Entra ID** authentication via `DefaultAzureCredential` for:

* **Azure AI Foundry** — OpenAI and Foundry Inference services (`https://cognitiveservices.azure.com/.default` scope). Requires the *Cognitive Services OpenAI User* or *Cognitive Services User* RBAC role.
* **Azure Blob Storage** — upload and read operations. Requires the *Storage Blob Data Contributor* RBAC role on the storage account.

Azure AI Search uses **API key** authentication.

### Config File

Non-secret configuration is in `config.yaml`. The defaults work out of the box:

```yaml
models:
  embedding_model: text-embedding-3-large
  llm_model: gpt-4o
  image_embedding_model: embed-v-4-0

search:
  semantic_weight: 0.4
  structural_weight: 0.15
  style_weight: 0.15
  image_weight: 0.2
  keyword_weight: 0.1

index:
  name: candidate-index
  vector_dimensions:
    semantic: 3072
    structural: 1024
    style: 512
    image: 1024
  hnsw:
    m: 4
    ef_construction: 400
    ef_search: 500

retrieval:
  top_k: 50
  k_nearest: 100

extraction:
  image_detail: high
  temperature: 0.2
  max_tokens: 4096

batch:
  index_batch_size: 500
  embedding_chunk_size: 2048
  max_concurrent_requests: 50
```

## Usage

Every command below assumes you have activated the venv and set the SSL cert path. Run these two lines once per terminal session:

```bash
source .venv/bin/activate
export SSL_CERT_FILE=/private/etc/ssl/cert.pem
```

> [!TIP]
> Add the `SSL_CERT_FILE` export to your `~/.zshrc` (or `~/.bashrc`) so it persists across sessions.

### Step 1: Create the Search Index

Create or update the Azure AI Search index with the schema defined in the project (20 fields: 16 primitive + 4 vector):

```bash
python -m ai_search.indexing.cli create
```

Or using the CLI entry point:

```bash
ai-search-index create
```

Expected output:

```text
Index 'candidate-index' created/updated successfully
```

### Step 2: Ingest Images into the Index

#### Option A: Batch Ingest Sample Images

The project includes 50 Ramayana-themed sample images (generated via `gpt-image-1.5`) defined in `data/sample_images.json` with their generation prompts. Ingest them all:

```bash
python scripts/ingest_samples.py
```

Flags:

| Flag              | Purpose                                         |
|-------------------|--------------------------------------------------|
| `--dry-run`       | Preview which images would be processed         |
| `--force`         | Re-ingest all images, even if already indexed   |
| `--concurrency N` | Number of concurrent processing tasks (default: 3) |

```bash
# Ingest with higher concurrency
python scripts/ingest_samples.py --force --concurrency 8
```

When `AZURE_STORAGE_ACCOUNT_NAME` is set, each image is automatically uploaded to Azure Blob Storage and the public blob URL is stored in the search index.

#### Option A.1: Retry Failed Images

If some images fail during batch ingestion (for example, due to content filters or transient errors), retry them:

```bash
python scripts/retry_failed.py
```

Edit the `FAILED_IDS` list inside the script to target specific image IDs.

#### Option B: Ingest a Single Image

From a URL:

```bash
ai-search-ingest \
  --image-url "https://example.com/photo.jpg" \
  --prompt "A cinematic night scene in Tokyo" \
  --image-id "my-image-001"
```

From a local file:

```bash
ai-search-ingest \
  --image-file ./path/to/photo.jpg \
  --prompt "A cinematic night scene in Tokyo" \
  --image-id "my-image-001"
```

| Argument       | Required | Purpose                              |
|----------------|----------|--------------------------------------|
| `--image-url`  | One of   | Public URL of the image to ingest    |
| `--image-file` | One of   | Local path to the image file         |
| `--prompt`     | **Yes**  | Generation prompt describing the image |
| `--image-id`   | **Yes**  | Unique identifier for the image      |

#### What Happens During Ingestion

Each image goes through this pipeline:

1. **Blob Upload** — if Azure Blob Storage is configured, the image is uploaded to the `images` container and a public blob URL is generated
2. **GPT-4o Extraction** — analyzes the image and produces structured metadata (descriptions, characters, emotion, objects, low-light metrics)
3. **Embedding Generation** — creates 4 vectors:
   * Semantic (3072d) from scene description via text-embedding-3-large
   * Structural (1024d) from composition description via text-embedding-3-large
   * Style (512d) from artistic style description via text-embedding-3-large
   * Image (1024d) from raw pixels via Cohere Embed v4 (`embed-v-4-0`)
4. **Document Upload** — uploads the document with all 20 fields (including `image_url` blob reference) to Azure AI Search

#### Ingestion Pipeline Module

The core ingestion logic lives in `src/ai_search/ingestion/pipeline.py` and can be used programmatically:

```python
import asyncio
from ai_search.ingestion.pipeline import ingest

result = asyncio.run(ingest(
    data_path="data/sample_images.json",
    images_dir="data/images",
    force=False,
    concurrency=3,
))

print(f"Processed: {result.processed}")
print(f"Skipped:   {result.skipped}")
print(f"Failed:    {result.failed}")
for image_id, error in result.errors.items():
    print(f"  {image_id}: {error}")
```

Key exports from the module:

| Function | Purpose |
|----------|---------|
| `ingest()` | Full orchestrator — load, upload, process, index |
| `process_image()` | Process a single image (extraction + embeddings) |
| `upload_to_blob()` | Upload image bytes to Azure Blob Storage |
| `load_image_inputs()` | Parse `sample_images.json` into `ImageInput` objects |
| `load_image_bytes()` | Read image files from disk |
| `get_already_indexed()` | Query existing image IDs from the search index |
| `index_documents()` | Batch-upload `SearchDocument` objects to the index |
| `IngestionResult` | Dataclass with `processed`, `skipped`, `failed`, `errors` fields |

### Step 3: Search the Index

The pipeline supports two search modes:

| Mode  | Input        | Strategy                                     | Scoring              |
|-------|--------------|----------------------------------------------|----------------------|
| Text  | Free-text    | 3-vector RRF fusion (semantic + structural + style) | Min-max normalized 0–1 |
| Image | Image upload | GPT-4o extraction + 4-vector RRF fusion (3 text vecs + image vec) | Min-max normalized 0–1 |

#### Text Search (CLI)

Run a multi-vector search query (3 vector searches fused via Reciprocal Rank Fusion):

```bash
ai-search-query \
  --query "cinematic night scene with neon lights" \
  --top 10
```

With an OData filter:

```bash
ai-search-query \
  --query "portrait photography" \
  --filter "scene_type eq 'portrait'" \
  --top 5
```

| Argument   | Required | Default | Purpose                       |
|------------|----------|---------|-------------------------------|
| `--query`  | **Yes**  | —       | Search query text             |
| `--top`    | No       | 10      | Number of results to return   |
| `--filter` | No       | —       | OData filter expression       |

Example output:

```text
============================================================
Query: Hanuman flying over the ocean
Results: 3
============================================================

  1. [ramayana-016] score=1.0000  scene=Mythological Flight
     url=https://<storage>.blob.core.windows.net/images/ramayana-016.png
     tags=['hanuman', 'ocean', 'flight', 'devotion']
  2. [ramayana-025] score=0.8342  scene=Divine Journey
     url=https://<storage>.blob.core.windows.net/images/ramayana-025.png
     tags=['aerial', 'ocean', 'mythological']
```

#### Text Search (Python)

Use the unified `search()` function in your own code:

```python
import asyncio
from ai_search.models import SearchMode
from ai_search.retrieval.pipeline import search

results = asyncio.run(search(
    mode=SearchMode.TEXT,
    query_text="golden palace coronation ceremony",
    top=5,
))

for r in results:
    print(f"{r.image_id}: {r.search_score:.4f} — {r.scene_type}")
    print(f"  url={r.image_url}")  # Azure Blob Storage URL
```

#### Image Search (Python)

Upload an image and find visually similar indexed images using GPT-4o extraction and 4-vector RRF:

```python
import asyncio
from pathlib import Path
from ai_search.models import SearchMode
from ai_search.retrieval.pipeline import search

image_bytes = Path("photo.jpg").read_bytes()

results = asyncio.run(search(
    mode=SearchMode.IMAGE,
    query_image_bytes=image_bytes,
    top=5,
))

for r in results:
    print(f"{r.image_id}: {r.search_score:.4f} — {r.scene_type}")
    print(f"  url={r.image_url}")
```

Image search works in three steps:

1. Embeds the query image via Cohere Embed v4 to produce an `image_vector`
2. Extracts semantic, structural, and style descriptions from the image via GPT-4o, then embeds each into its corresponding text vector space via text-embedding-3-large
3. Sends all 4 vectors (`semantic_vector`, `structural_vector`, `style_vector`, `image_vector`) to Azure AI Search, which fuses results via Reciprocal Rank Fusion (RRF) and returns the top K results with min-max normalized scores

Search results include `image_url` pointing to the Azure Blob Storage URL (for example, `https://<account>.blob.core.windows.net/images/ramayana-001.png`) when images were ingested with blob storage enabled.

#### Visual Search UI (Gradio)

Launch the web UI to search interactively with both modes in a browser:

```bash
source .venv/bin/activate
export SSL_CERT_FILE=/private/etc/ssl/cert.pem
python -m ai_search.ui.app
```

Or using the CLI entry point:

```bash
ai-search-ui
```

Open <http://localhost:7860> in your browser. The UI provides two tabs:

* **Text → Image** — type a query, get ranked image results with scores
* **Image → Image** — upload an image, find visually similar indexed images

Both tabs display a results gallery with score bars, image IDs, scene types, tags, and generation prompts.

#### Search Test Scripts

Test scripts are available in `scripts/`:

```bash
# Text + image search smoke test (text queries + image-to-image)
python scripts/test_search.py

# Image-to-image similarity search
python scripts/test_image_search.py

# Compare hybrid vs image-only search strategies
python scripts/test_hybrid_vs_image.py

# Relevance scoring with confidence tiers
python scripts/test_relevance_tiers.py

# Test Cohere Embed v4 image embedding directly
python scripts/test_image_embed.py

# Test Foundry Inference SDK connectivity
python scripts/test_inference_sdk.py
```

## Development

### Running Tests

```bash
source .venv/bin/activate

# Run all unit tests (50 tests)
PYTHONPATH=src python -m pytest

# Run with coverage
PYTHONPATH=src python -m pytest --cov=ai_search

# Run a specific test module
PYTHONPATH=src python -m pytest tests/test_embeddings/test_image.py -v

# Skip integration tests
PYTHONPATH=src python -m pytest -m "not integration"
```

### Linting and Type Checking

```bash
# Lint with ruff
ruff check src/ tests/

# Type check with mypy
mypy src/
```

### Project Structure

```text
ai-search/
├── config.yaml              # Non-secret configuration
├── pyproject.toml            # Dependencies and entry points
├── data/
│   ├── images/               # 50 Ramayana-themed sample images
│   └── sample_images.json    # Image metadata and generation prompts
├── docs/
│   ├── architecture.md       # Full architecture documentation
│   ├── image-indexing-process.md
│   ├── search-process.md
│   └── learnings-image-embedding-fix.md
├── scripts/
│   ├── setup_azure.sh        # Automated Azure resource provisioning
│   ├── generate_ramayana_images.py  # Generate images via gpt-image-1.5
│   ├── ingest_samples.py     # Batch ingest CLI wrapper
│   ├── retry_failed.py       # Retry failed ingestions
│   ├── test_search.py        # Text + image search tests
│   ├── test_image_search.py  # Image-to-image similarity tests
│   ├── test_hybrid_vs_image.py
│   └── test_relevance_tiers.py
├── src/ai_search/
│   ├── clients.py            # Azure SDK client factories
│   ├── config.py             # Configuration and secrets
│   ├── models.py             # Shared data models (SearchMode, SearchResult, etc.)
│   ├── embeddings/           # Text + image embedding pipeline
│   ├── extraction/           # GPT-4o image analysis
│   ├── indexing/             # Index schema + document upload
│   ├── ingestion/            # Ingestion pipeline + image input + CLI
│   │   ├── cli.py            # Single-image CLI entry point
│   │   ├── loader.py         # Image loading utilities
│   │   ├── metadata.py       # Metadata handling
│   │   └── pipeline.py       # Reusable ingestion orchestrator
│   ├── retrieval/            # Unified search pipeline + relevance scoring
│   ├── storage/              # Cloud storage integration
│   │   └── blob.py           # Azure Blob Storage upload utilities
│   └── ui/                   # Gradio web UI (text + image search)
└── tests/                    # Unit and integration tests
```

For detailed architecture documentation including API specifications, SDK classes, index schema details, and HNSW configuration, see [docs/architecture.md](docs/architecture.md).

## Architecture Overview

```text
┌─────────────────── INGESTION ───────────────────┐
│                                                 │
│  Image Input                                    │
│      │                                          │
│      ├──► Azure Blob Storage (cloud backup)     │
│      │                                          │
│      ▼                                          │
│  GPT-4o Extraction ──► ImageExtraction           │
│      │                                          │
│      ▼                                          │
│  Embedding Pipeline                             │
│      ├── text-embedding-3-large ──► 3 text vecs │
│      └── embed-v-4-0            ──► image vec   │
│      │                                          │
│      ▼                                          │
│  Azure AI Search ──► SearchDocument (20 fields)  │
│       (includes blob URL in image_url field)     │
│                                                 │
└─────────────────────────────────────────────────┘

┌─────────────────── RETRIEVAL ───────────────────┐
│                                                 │
│  search(mode=SearchMode.TEXT)                    │
│      ├── LLM query expansion (structural+style) │
│      ├── 3× text vector embedding                │
│      ├── Multi-vector RRF fusion (no BM25)       │
│      └── Min-max normalized scores (0–1)         │
│                                                 │
│  search(mode=SearchMode.IMAGE)                   │
│      ├── GPT-4o extraction → 3 descriptions      │
│      ├── 3× text embedding + 1× image embedding  │
│      ├── 4-vector RRF fusion via Azure AI Search │
│      └── Min-max normalized scores (0–1)         │
│                                                 │
│                      ▼                           │
│              SearchResult[]                      │
│                                                 │
└─────────────────────────────────────────────────┘
```