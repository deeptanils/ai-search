# AI Search Pipeline

Candidate Generation & AI Search Pipeline using Azure AI Foundry and Azure AI Search. Accepts images with generation prompts, extracts structured metadata via GPT-4o, generates multi-vector embeddings, indexes into Azure AI Search, and supports hybrid multi-vector retrieval with relevance scoring.

## Prerequisites

* Python 3.11+
* [UV](https://docs.astral.sh/uv/) package manager
* Azure subscription with the following resources provisioned:
  * **Azure AI Foundry** — GPT-4o and text-embedding-3-large deployments, plus Cohere Embed v4 (`embed-v-4-0`) serverless deployment
  * **Azure AI Search** — Basic tier or higher
  * **Azure Computer Vision** (optional) — only required if using Florence backend instead of Foundry for image embeddings

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

# Azure Computer Vision (optional — only for Florence backend)
AZURE_CV_ENDPOINT=https://your-cv-resource.cognitiveservices.azure.com
AZURE_CV_API_KEY=your-cv-api-key-here
```

| Variable                       | Required | Purpose                                      |
|--------------------------------|----------|----------------------------------------------|
| `AZURE_FOUNDRY_ENDPOINT`       | **Yes**  | Azure AI Foundry base endpoint (OpenAI API)  |
| `AZURE_FOUNDRY_EMBED_ENDPOINT` | No       | Foundry models endpoint for Inference SDK    |
| `AZURE_FOUNDRY_API_KEY`        | No       | API key fallback (Entra ID preferred)        |
| `AZURE_OPENAI_API_VERSION`     | No       | OpenAI API version (default: `2024-12-01-preview`) |
| `AZURE_AI_SEARCH_ENDPOINT`     | **Yes**  | Azure AI Search service endpoint             |
| `AZURE_AI_SEARCH_API_KEY`      | **Yes**  | Azure AI Search admin API key                |
| `AZURE_AI_SEARCH_INDEX_NAME`   | No       | Index name (default: `candidate-index`)      |
| `AZURE_CV_ENDPOINT`            | No       | Computer Vision endpoint (Florence only)     |
| `AZURE_CV_API_KEY`             | No       | Computer Vision API key (Florence only)      |

### Authentication

The pipeline uses **Entra ID** authentication for Azure OpenAI and Foundry Inference services via `DefaultAzureCredential` with the scope `https://cognitiveservices.azure.com/.default`. Ensure your Azure identity (user or service principal) has the appropriate RBAC roles on the Foundry resource.

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

### Step 1: Create the Search Index

Create or update the Azure AI Search index with the schema defined in the project (19 fields: 15 primitive + 4 vector):

```bash
source .venv/bin/activate
SSL_CERT_FILE=/private/etc/ssl/cert.pem PYTHONPATH=src python -m ai_search.indexing.cli create
```

Or using the CLI entry point:

```bash
SSL_CERT_FILE=/private/etc/ssl/cert.pem ai-search-index create
```

### Step 2: Ingest Images into the Index

#### Option A: Batch Ingest Sample Images

The project includes 10 sample images in `data/sample_images.json`. Ingest them all:

```bash
SSL_CERT_FILE=/private/etc/ssl/cert.pem PYTHONPATH=src python scripts/ingest_samples.py
```

Flags:

| Flag        | Purpose                                         |
|-------------|--------------------------------------------------|
| `--dry-run` | Preview which images would be processed         |
| `--force`   | Re-ingest all images, even if already indexed   |

The script introduces a 10-second delay between images to respect Foundry S0 rate limits. On HTTP 429 errors, it retries up to 5 times with 30-second backoff.

#### Option B: Ingest a Single Image

From a URL:

```bash
SSL_CERT_FILE=/private/etc/ssl/cert.pem ai-search-ingest \
  --image-url "https://example.com/photo.jpg" \
  --prompt "A cinematic night scene in Tokyo" \
  --image-id "my-image-001"
```

From a local file:

```bash
SSL_CERT_FILE=/private/etc/ssl/cert.pem ai-search-ingest \
  --image-file ./path/to/photo.jpg \
  --prompt "A cinematic night scene in Tokyo" \
  --image-id "my-image-001"
```

#### What Happens During Ingestion

Each image goes through this pipeline:

1. **GPT-4o Extraction** — analyzes the image and produces structured metadata (descriptions, characters, emotion, objects, low-light metrics)
2. **Embedding Generation** — creates 4 vectors:
   * Semantic (3072d) from scene description via text-embedding-3-large
   * Structural (1024d) from composition description via text-embedding-3-large
   * Style (512d) from artistic style description via text-embedding-3-large
   * Image (1024d) from raw pixels via Cohere Embed v4 (`embed-v-4-0`)
3. **Document Upload** — uploads the document with all fields to Azure AI Search

### Step 3: Search the Index

Run a hybrid search query (BM25 + 4 vector searches fused via RRF):

```bash
SSL_CERT_FILE=/private/etc/ssl/cert.pem ai-search-query \
  --query "cinematic night scene with neon lights" \
  --top 10
```

With an OData filter:

```bash
SSL_CERT_FILE=/private/etc/ssl/cert.pem ai-search-query \
  --query "portrait photography" \
  --filter "scene_type eq 'portrait'" \
  --top 5
```

#### Search Scripts

Additional search test scripts are available in `scripts/`:

```bash
# Basic keyword and filter search
SSL_CERT_FILE=/private/etc/ssl/cert.pem PYTHONPATH=src python scripts/test_search.py

# Image-to-image similarity search
SSL_CERT_FILE=/private/etc/ssl/cert.pem PYTHONPATH=src python scripts/test_image_search.py

# Compare hybrid vs image-only search
SSL_CERT_FILE=/private/etc/ssl/cert.pem PYTHONPATH=src python scripts/test_hybrid_vs_image.py

# Relevance scoring with confidence tiers
SSL_CERT_FILE=/private/etc/ssl/cert.pem PYTHONPATH=src python scripts/test_relevance_tiers.py
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
│   └── sample_images.json    # 10 sample images for testing
├── docs/
│   ├── architecture.md       # Full architecture documentation
│   └── learnings-image-embedding-fix.md  # Post-mortem and learnings
├── scripts/                  # Utility and test scripts
├── src/ai_search/
│   ├── clients.py            # Azure SDK client factories
│   ├── config.py             # Configuration and secrets
│   ├── models.py             # Shared data models
│   ├── embeddings/           # Text + image embedding pipeline
│   ├── extraction/           # GPT-4o image analysis
│   ├── indexing/             # Index schema + document upload
│   ├── ingestion/            # Image input + CLI
│   └── retrieval/            # Hybrid search + relevance scoring
└── tests/                    # Unit and integration tests
```

For detailed architecture documentation including API specifications, SDK classes, index schema details, and HNSW configuration, see [docs/architecture.md](docs/architecture.md).

## Architecture Overview

```text
Image Input
    │
    ▼
GPT-4o Extraction ──► ImageExtraction (descriptions, metadata, characters)
    │
    ▼
Embedding Pipeline
    ├── text-embedding-3-large ──► semantic (3072d), structural (1024d), style (512d)
    └── embed-v-4-0            ──► image (1024d)
    │
    ▼
Azure AI Search ──► SearchDocument (19 fields) ──► Index Upload
    │
    ▼
Hybrid Retrieval
    ├── BM25 full-text search
    ├── 4× vector search (weighted RRF fusion)
    └── Relative relevance scoring (HIGH / MEDIUM / LOW confidence)
```