<!-- markdownlint-disable-file -->
# Implementation Details: Candidate Generation & AI Search Pipeline

## Context Reference

Sources: [ai-search-pipeline-research.md](../../research/2026-02-26/ai-search-pipeline-research.md), [azure-ai-foundry-sdk-research.md](../../research/subagents/2026-02-26/azure-ai-foundry-sdk-research.md), [azure-ai-search-sdk-research.md](../../research/subagents/2026-02-26/azure-ai-search-sdk-research.md), [uv-python-project-research.md](../../research/subagents/2026-02-26/uv-python-project-research.md), [multi-vector-encoding-research.md](../../research/subagents/2026-02-26/multi-vector-encoding-research.md), [hybrid-retrieval-research.md](../../research/subagents/2026-02-26/hybrid-retrieval-research.md)

## Implementation Phase 1: Project Scaffolding

<!-- parallelizable: false -->

### Step 1.1: Initialize UV project and create directory structure

Run UV init and create the full directory tree for the project.

Commands:
```bash
cd /Users/deesaha/Desktop/WORK/projects/ai-search
uv init --app --name ai-search
uv python pin 3.12
mkdir -p src/ai_search/{ingestion,extraction,embeddings,indexing,retrieval}
mkdir -p tests/{test_ingestion,test_extraction,test_embeddings,test_indexing,test_retrieval}
```

Files:
* `src/ai_search/` - Top-level importable package
* `src/ai_search/ingestion/` - Image loading and LLM metadata generation
* `src/ai_search/extraction/` - GPT-4o vision extraction (narrative, emotion, objects, low-light)
* `src/ai_search/embeddings/` - Multi-vector embedding generation
* `src/ai_search/indexing/` - Azure AI Search index schema and document upload
* `src/ai_search/retrieval/` - Hybrid search, re-ranking, and MMR diversity
* `tests/` - Test directory mirroring src structure

Success criteria:
* Directory tree matches requirements.md Section 11
* UV creates `.python-version` with 3.12

Context references:
* uv-python-project-research.md (Lines 20-50) - UV init commands
* uv-python-project-research.md (Lines 170-220) - Directory layout

Dependencies:
* UV installed on system

### Step 1.2: Create pyproject.toml with all dependencies and tooling config

Create the complete `pyproject.toml` with project metadata, runtime dependencies, dev dependency groups, CLI entry points, hatchling build config, ruff, mypy, and pytest settings.

Files:
* `pyproject.toml` - Project configuration

Content:
```toml
[project]
name = "ai-search"
version = "0.1.0"
description = "Candidate Generation & AI Search Pipeline using Azure AI Foundry and Azure AI Search"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }

dependencies = [
    "openai>=1.58.0",
    "azure-search-documents>=11.6.0",
    "azure-identity>=1.17.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "pyyaml>=6.0",
    "python-dotenv>=1.0",
    "pillow>=10.0",
    "httpx>=0.27",
    "structlog>=24.0",
    "numpy>=1.26",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-cov>=5.0",
    "ruff>=0.5",
    "mypy>=1.10",
    "types-PyYAML>=6.0",
    "types-Pillow>=10.0",
]

[project.scripts]
ai-search-ingest = "ai_search.ingestion.cli:main"
ai-search-index = "ai_search.indexing.cli:main"
ai-search-query = "ai_search.retrieval.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/ai_search"]

[tool.ruff]
target-version = "py311"
line-length = 120
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B", "SIM", "TCH"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true
mypy_path = "src"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "-v --tb=short --strict-markers"
asyncio_mode = "auto"
markers = [
    "integration: marks tests as integration tests (deselect with '-m \"not integration\"')",
    "slow: marks tests as slow",
]
```

Success criteria:
* `uv sync` installs all dependencies without errors
* hatchling build config maps `src/ai_search/` correctly

Context references:
* uv-python-project-research.md (Lines 70-160) - pyproject.toml structure
* uv-python-project-research.md (Lines 100-120) - dependency groups (PEP 735)

Dependencies:
* Step 1.1 completion (UV project initialized)

### Step 1.3: Create config.yaml, .env.example, .gitignore, and README.md

Create all configuration and project files.

Files:
* `config.yaml` - Non-secret configuration
* `.env.example` - Secret template (committed to git, no real secrets)
* `.gitignore` - Git ignore patterns
* `README.md` - Project documentation

config.yaml content:
```yaml
models:
  embedding_model: text-embedding-3-large
  llm_model: gpt-4o

search:
  semantic_weight: 0.5
  structural_weight: 0.2
  style_weight: 0.2
  keyword_weight: 0.1

index:
  name: candidate-index
  vector_dimensions:
    semantic: 3072
    structural: 1024
    style: 512
    character_semantic: 512
    character_emotion: 256
    character_pose: 256
  hnsw:
    m: 4
    ef_construction: 400
    ef_search: 500
  max_character_slots: 3

retrieval:
  stage1_top_k: 200
  stage1_k_nearest: 100
  stage2_top_k: 50
  stage3_top_k: 20
  mmr_lambda: 0.6
  rerank_weights:
    emotional: 0.3
    narrative: 0.25
    object_overlap: 0.25
    low_light: 0.2

extraction:
  image_detail: high
  temperature: 0.2
  max_tokens: 4096

batch:
  index_batch_size: 500
  embedding_chunk_size: 2048
  max_concurrent_requests: 50
```

.env.example content:
```env
# Azure AI Foundry
AZURE_FOUNDRY_ENDPOINT=https://your-foundry.cognitiveservices.azure.com/
AZURE_FOUNDRY_API_KEY=your-api-key-here
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# Azure AI Search
AZURE_AI_SEARCH_ENDPOINT=https://your-search.search.windows.net
AZURE_AI_SEARCH_API_KEY=your-search-key-here
AZURE_AI_SEARCH_INDEX_NAME=candidate-index
```

.gitignore content:
```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/

# Virtual environment
.venv/

# Secrets
.env

# IDE
.vscode/
.idea/

# OS
.DS_Store

# Testing
.coverage
htmlcov/
.pytest_cache/

# Mypy
.mypy_cache/

# Ruff
.ruff_cache/
```

Success criteria:
* config.yaml loads with PyYAML without errors
* .env.example documents all required environment variables
* .gitignore excludes .env, .venv, __pycache__

Context references:
* uv-python-project-research.md (Lines 290-400) - Config management pattern
* requirements.md Section 3 - .env and config.yaml specifications

Dependencies:
* Step 1.1 completion

### Step 1.4: Create all `__init__.py` files for package structure

Create `__init__.py` in every package directory. Each file includes a module docstring.

Files:
* `src/ai_search/__init__.py` - Package root with `__version__`
* `src/ai_search/ingestion/__init__.py`
* `src/ai_search/extraction/__init__.py`
* `src/ai_search/embeddings/__init__.py`
* `src/ai_search/indexing/__init__.py`
* `src/ai_search/retrieval/__init__.py`

`src/ai_search/__init__.py` content:
```python
"""Candidate Generation & AI Search Pipeline."""

__version__ = "0.1.0"
```

Each subpackage `__init__.py` content (example for ingestion):
```python
"""Ingestion module — loads images and generates synthetic metadata."""
```

Success criteria:
* `from ai_search import __version__` works after `uv sync`
* All subpackages are importable

Context references:
* uv-python-project-research.md (Lines 225-260) - __init__.py patterns

Dependencies:
* Step 1.1 completion (directories exist)

### Step 1.5: Run `uv sync` and verify import works

Install all dependencies and verify the package is importable.

Commands:
```bash
uv sync
uv run python -c "import ai_search; print(ai_search.__version__)"
uv run pytest --co  # collect tests (dry run)
```

Success criteria:
* `uv sync` completes without errors
* `import ai_search` prints "0.1.0"
* `uv.lock` is generated

Dependencies:
* Steps 1.1-1.4 completed

## Implementation Phase 2: Configuration & Shared Models

<!-- parallelizable: false -->

### Step 2.1: Create config.py with pydantic-settings + YAML loader

Create the configuration module that loads secrets from `.env` via pydantic-settings and non-secret config from `config.yaml` via PyYAML.

Files:
* `src/ai_search/config.py` - Configuration management

Implementation:
```python
"""Configuration management — loads .env secrets and config.yaml settings."""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AzureFoundrySecrets(BaseSettings):
    """Azure AI Foundry secrets from .env."""

    model_config = SettingsConfigDict(
        env_prefix="AZURE_FOUNDRY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    endpoint: str
    api_key: str


class AzureOpenAISecrets(BaseSettings):
    """Azure OpenAI API version from .env."""

    model_config = SettingsConfigDict(
        env_prefix="AZURE_OPENAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_version: str = "2024-12-01-preview"


class AzureSearchSecrets(BaseSettings):
    """Azure AI Search secrets from .env."""

    model_config = SettingsConfigDict(
        env_prefix="AZURE_AI_SEARCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    endpoint: str
    api_key: str
    index_name: str = "candidate-index"


class ModelsConfig(BaseModel):
    """Model deployment names."""

    embedding_model: str = "text-embedding-3-large"
    llm_model: str = "gpt-4o"


class SearchWeightsConfig(BaseModel):
    """Retrieval weight configuration."""

    semantic_weight: float = 0.5
    structural_weight: float = 0.2
    style_weight: float = 0.2
    keyword_weight: float = 0.1


class VectorDimensionsConfig(BaseModel):
    """Vector dimension configuration."""

    semantic: int = 3072
    structural: int = 1024
    style: int = 512
    character_semantic: int = 512
    character_emotion: int = 256
    character_pose: int = 256


class HnswConfig(BaseModel):
    """HNSW algorithm parameters."""

    m: int = 4
    ef_construction: int = 400
    ef_search: int = 500


class IndexConfig(BaseModel):
    """Index configuration."""

    name: str = "candidate-index"
    vector_dimensions: VectorDimensionsConfig = VectorDimensionsConfig()
    hnsw: HnswConfig = HnswConfig()
    max_character_slots: int = 3


class RerankWeightsConfig(BaseModel):
    """Re-ranking weight configuration."""

    emotional: float = 0.3
    narrative: float = 0.25
    object_overlap: float = 0.25
    low_light: float = 0.2


class RetrievalConfig(BaseModel):
    """Retrieval pipeline configuration."""

    stage1_top_k: int = 200
    stage1_k_nearest: int = 100
    stage2_top_k: int = 50
    stage3_top_k: int = 20
    mmr_lambda: float = 0.6
    rerank_weights: RerankWeightsConfig = RerankWeightsConfig()


class ExtractionConfig(BaseModel):
    """GPT-4o extraction configuration."""

    image_detail: str = "high"
    temperature: float = 0.2
    max_tokens: int = 4096


class BatchConfig(BaseModel):
    """Batch processing configuration."""

    index_batch_size: int = 500
    embedding_chunk_size: int = 2048
    max_concurrent_requests: int = 50


class AppConfig(BaseModel):
    """Full application configuration from config.yaml."""

    models: ModelsConfig = ModelsConfig()
    search: SearchWeightsConfig = SearchWeightsConfig()
    index: IndexConfig = IndexConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    extraction: ExtractionConfig = ExtractionConfig()
    batch: BatchConfig = BatchConfig()


@lru_cache(maxsize=1)
def load_config(config_path: Path = Path("config.yaml")) -> AppConfig:
    """Load non-secret configuration from config.yaml."""
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        return AppConfig(**data)
    return AppConfig()


@lru_cache(maxsize=1)
def load_foundry_secrets() -> AzureFoundrySecrets:
    """Load Azure AI Foundry secrets from .env."""
    return AzureFoundrySecrets()  # type: ignore[call-arg]


@lru_cache(maxsize=1)
def load_openai_secrets() -> AzureOpenAISecrets:
    """Load Azure OpenAI API version from .env."""
    return AzureOpenAISecrets()  # type: ignore[call-arg]


@lru_cache(maxsize=1)
def load_search_secrets() -> AzureSearchSecrets:
    """Load Azure AI Search secrets from .env."""
    return AzureSearchSecrets()  # type: ignore[call-arg]
```

Success criteria:
* Config loads from yaml with default fallbacks
* Secrets load from .env via pydantic-settings
* All config classes validate their fields

Context references:
* uv-python-project-research.md (Lines 300-380) - Pydantic settings pattern
* azure-ai-foundry-sdk-research.md (Lines 38-70) - Client initialization pattern

Dependencies:
* Phase 1 complete

### Step 2.2: Create models.py with shared Pydantic data models

Define all Pydantic models for pipeline data flow: image input, extraction output, metadata, character descriptions, and search documents.

Files:
* `src/ai_search/models.py` - Shared data models

Implementation:
```python
"""Shared Pydantic models for the AI Search pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CharacterDescription(BaseModel):
    """Per-character structured descriptions for embedding."""

    character_id: str = Field(description="Descriptive identifier, e.g. 'woman_red_dress'")
    semantic: str = Field(description="2-3 sentences: identity, role, appearance, clothing")
    emotion: str = Field(description="2-3 sentences: emotional expression, body language cues")
    pose: str = Field(description="2-3 sentences: physical position, orientation, gestures")


class ImageMetadata(BaseModel):
    """Synthetic metadata generated by LLM."""

    scene_type: str
    time_of_day: str
    lighting_condition: str
    primary_subject: str
    secondary_subjects: list[str] = Field(default_factory=list)
    artistic_style: str
    color_palette: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    narrative_theme: str


class NarrativeIntent(BaseModel):
    """Narrative analysis of the image."""

    story_summary: str
    narrative_type: str  # cinematic, documentary, surreal, fantasy, etc.
    tone: str


class EmotionalTrajectory(BaseModel):
    """Emotional trajectory analysis."""

    starting_emotion: str
    mid_emotion: str
    end_emotion: str
    emotional_polarity: float = Field(ge=-1.0, le=1.0)


class RequiredObjects(BaseModel):
    """Objects detected in the scene."""

    key_objects: list[str] = Field(default_factory=list)
    contextual_objects: list[str] = Field(default_factory=list)
    symbolic_elements: list[str] = Field(default_factory=list)


class LowLightMetrics(BaseModel):
    """Low-light robustness indicators."""

    brightness_score: float = Field(ge=0.0, le=1.0)
    contrast_score: float = Field(ge=0.0, le=1.0)
    noise_estimate: float = Field(ge=0.0, le=1.0)
    shadow_dominance: float = Field(ge=0.0, le=1.0)
    visibility_confidence: float = Field(ge=0.0, le=1.0)


class ImageExtraction(BaseModel):
    """Complete GPT-4o vision extraction output (used as structured output schema)."""

    semantic_description: str = Field(description="Rich 200-word description for semantic embedding")
    structural_description: str = Field(description="150-word spatial/composition analysis for structural embedding")
    style_description: str = Field(description="150-word artistic style analysis for style embedding")
    characters: list[CharacterDescription] = Field(default_factory=list)
    metadata: ImageMetadata
    narrative: NarrativeIntent
    emotion: EmotionalTrajectory
    objects: RequiredObjects
    low_light: LowLightMetrics


class CharacterVectors(BaseModel):
    """Embedding vectors for a single character."""

    character_id: str
    semantic_vector: list[float]
    emotion_vector: list[float]
    pose_vector: list[float]


class ImageVectors(BaseModel):
    """All embedding vectors for an image."""

    semantic_vector: list[float]
    structural_vector: list[float]
    style_vector: list[float]
    character_vectors: list[CharacterVectors] = Field(default_factory=list)


class SearchDocument(BaseModel):
    """Document structure for Azure AI Search index upload."""

    image_id: str
    generation_prompt: str
    scene_type: str
    time_of_day: str
    lighting_condition: str
    primary_subject: str
    artistic_style: str
    tags: list[str]
    narrative_theme: str
    narrative_type: str
    emotional_polarity: float
    low_light_score: float
    character_count: int
    metadata_json: str  # Full metadata as JSON string for retrieval
    extraction_json: str  # Full extraction as JSON string for re-ranking

    # Vectors (populated by embedding pipeline)
    semantic_vector: list[float] = Field(default_factory=list)
    structural_vector: list[float] = Field(default_factory=list)
    style_vector: list[float] = Field(default_factory=list)

    # Flattened character vectors (up to 3 slots)
    char_0_semantic_vector: list[float] = Field(default_factory=list)
    char_0_emotion_vector: list[float] = Field(default_factory=list)
    char_0_pose_vector: list[float] = Field(default_factory=list)
    char_1_semantic_vector: list[float] = Field(default_factory=list)
    char_1_emotion_vector: list[float] = Field(default_factory=list)
    char_1_pose_vector: list[float] = Field(default_factory=list)
    char_2_semantic_vector: list[float] = Field(default_factory=list)
    char_2_emotion_vector: list[float] = Field(default_factory=list)
    char_2_pose_vector: list[float] = Field(default_factory=list)


class QueryContext(BaseModel):
    """Context for re-ranking queries."""

    query_text: str | None = None
    emotions: EmotionalTrajectory | None = None
    narrative_intent: str | None = None
    required_objects: list[str] = Field(default_factory=list)
    low_light_score: float | None = None


class SearchResult(BaseModel):
    """Single search result with scores."""

    image_id: str
    search_score: float
    rerank_score: float | None = None
    generation_prompt: str | None = None
    scene_type: str | None = None
    tags: list[str] = Field(default_factory=list)
```

Success criteria:
* All models validate with strict type checking
* `ImageExtraction` works as `response_format` with `client.chat.completions.parse()`
* `SearchDocument` maps directly to Azure AI Search index fields

Context references:
* multi-vector-encoding-research.md (Lines 300-420) - Character vector structure
* requirements.md Section 3.2 - Synthetic metadata fields
* requirements.md Sections 4.1.1-4.1.6 - Extracted dimensions

Dependencies:
* Step 2.1 completion (config module available)

### Step 2.3: Create Azure AI Foundry client factory (sync + async)

Create a client factory module that initializes `AzureOpenAI` and `AsyncAzureOpenAI` clients from configuration.

Files:
* `src/ai_search/clients.py` - Client factories for Azure AI Foundry and Azure AI Search

Implementation:
```python
"""Client factories for Azure AI Foundry and Azure AI Search."""

from __future__ import annotations

from functools import lru_cache

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from openai import AsyncAzureOpenAI, AzureOpenAI

from ai_search.config import (
    load_foundry_secrets,
    load_openai_secrets,
    load_search_secrets,
)


@lru_cache(maxsize=1)
def get_openai_client() -> AzureOpenAI:
    """Return a cached synchronous Azure OpenAI client."""
    secrets = load_foundry_secrets()
    api = load_openai_secrets()
    return AzureOpenAI(
        azure_endpoint=secrets.endpoint,
        api_key=secrets.api_key,
        api_version=api.api_version,
    )


@lru_cache(maxsize=1)
def get_async_openai_client() -> AsyncAzureOpenAI:
    """Return a cached asynchronous Azure OpenAI client."""
    secrets = load_foundry_secrets()
    api = load_openai_secrets()
    return AsyncAzureOpenAI(
        azure_endpoint=secrets.endpoint,
        api_key=secrets.api_key,
        api_version=api.api_version,
    )


@lru_cache(maxsize=1)
def get_search_index_client() -> SearchIndexClient:
    """Return a cached Azure AI Search index management client."""
    secrets = load_search_secrets()
    return SearchIndexClient(
        endpoint=secrets.endpoint,
        credential=AzureKeyCredential(secrets.api_key),
    )


def get_search_client(index_name: str | None = None) -> SearchClient:
    """Return an Azure AI Search document operations client."""
    secrets = load_search_secrets()
    name = index_name or secrets.index_name
    return SearchClient(
        endpoint=secrets.endpoint,
        index_name=name,
        credential=AzureKeyCredential(secrets.api_key),
    )
```

Discrepancy references:
* DD-01: Research recommended `azure-ai-inference` as an alternative. Plan uses `openai` SDK exclusively per research recommendation.

Success criteria:
* Clients initialize using only `.env` secrets
* No secrets appear in code
* Async client supports `await client.embeddings.create(dimensions=...)`

Context references:
* azure-ai-foundry-sdk-research.md (Lines 38-70) - AzureOpenAI initialization
* azure-ai-search-sdk-research.md (Lines 30-60) - SearchClient initialization

Dependencies:
* Step 2.1 completion (config module)

### Step 2.4: Create Azure AI Search client factory

Included in Step 2.3 (clients.py contains both OpenAI and Search client factories).

Success criteria:
* `get_search_index_client()` returns `SearchIndexClient`
* `get_search_client()` returns `SearchClient` with configurable index name

Dependencies:
* Step 2.1 completion

## Implementation Phase 3: Ingestion & Extraction

<!-- parallelizable: true -->

### Step 3.1: Create ingestion/loader.py — image loading (URL + binary)

Image loading utilities that handle both URL-based and binary image inputs, encoding to base64 when needed.

Files:
* `src/ai_search/ingestion/loader.py` - Image loading utilities

Implementation:
```python
"""Image loading utilities for URL and binary inputs."""

from __future__ import annotations

import base64
from pathlib import Path

from pydantic import BaseModel


class ImageInput(BaseModel):
    """Validated image input."""

    image_id: str
    generation_prompt: str
    image_url: str | None = None
    image_base64: str | None = None

    @classmethod
    def from_url(cls, image_id: str, prompt: str, url: str) -> ImageInput:
        return cls(image_id=image_id, generation_prompt=prompt, image_url=url)

    @classmethod
    def from_file(cls, image_id: str, prompt: str, path: str | Path) -> ImageInput:
        data = Path(path).read_bytes()
        b64 = base64.standard_b64encode(data).decode("utf-8")
        return cls(image_id=image_id, generation_prompt=prompt, image_base64=b64)

    def to_openai_image_content(self) -> dict:
        """Create OpenAI image_url content part."""
        if self.image_url:
            return {"type": "image_url", "image_url": {"url": self.image_url, "detail": "high"}}
        if self.image_base64:
            url = f"data:image/jpeg;base64,{self.image_base64}"
            return {"type": "image_url", "image_url": {"url": url, "detail": "high"}}
        msg = "Either image_url or image_base64 must be set"
        raise ValueError(msg)
```

Success criteria:
* Loads images from URL or file path
* Produces OpenAI-compatible image content parts
* Validates that at least one image source is provided

Context references:
* azure-ai-foundry-sdk-research.md (Lines 300-350) - Image input patterns

Dependencies:
* Phase 2 complete

### Step 3.2: Create ingestion/metadata.py — LLM synthetic metadata generation

Standalone LLM metadata generation using GPT-4o with structured output (for the case where metadata is generated separately from full extraction).

Files:
* `src/ai_search/ingestion/metadata.py` - Synthetic metadata generation

Implementation:
```python
"""Synthetic metadata generation via GPT-4o."""

from __future__ import annotations

from ai_search.clients import get_openai_client
from ai_search.config import load_config
from ai_search.ingestion.loader import ImageInput
from ai_search.models import ImageMetadata

METADATA_SYSTEM_PROMPT = """You are a metadata extraction assistant. Given an image and its generation prompt, extract structured metadata. Be specific and accurate."""


def generate_metadata(image_input: ImageInput) -> ImageMetadata:
    """Generate synthetic metadata for an image using GPT-4o."""
    config = load_config()
    client = get_openai_client()

    response = client.beta.chat.completions.parse(
        model=config.models.llm_model,
        messages=[
            {"role": "system", "content": METADATA_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Generation prompt: {image_input.generation_prompt}\n\nExtract metadata.",
                    },
                    image_input.to_openai_image_content(),
                ],
            },
        ],
        response_format=ImageMetadata,
        temperature=config.extraction.temperature,
        max_tokens=1000,
    )

    parsed = response.choices[0].message.parsed
    if parsed is None:
        msg = "GPT-4o returned no parsed metadata"
        raise ValueError(msg)
    return parsed
```

Success criteria:
* Returns validated `ImageMetadata` Pydantic model
* Uses structured output (schema-enforced JSON)
* Handles both URL and base64 images

Context references:
* azure-ai-foundry-sdk-research.md (Lines 130-180) - Structured outputs with Pydantic
* requirements.md Section 5 - Metadata fields

Dependencies:
* Steps 2.1-2.3 complete (config, models, clients)

### Step 3.3: Create extraction module — unified GPT-4o vision extraction

The core extraction function that makes a single GPT-4o vision call to extract all descriptions, metadata, narrative, emotion, objects, and low-light metrics.

Files:
* `src/ai_search/extraction/extractor.py` - Unified GPT-4o vision extraction

Implementation:
```python
"""Unified GPT-4o vision extraction for all pipeline dimensions."""

from __future__ import annotations

from ai_search.clients import get_openai_client
from ai_search.config import load_config
from ai_search.ingestion.loader import ImageInput
from ai_search.models import ImageExtraction

EXTRACTION_SYSTEM_PROMPT = """You are an image analysis system. Given an image and the prompt that generated it, extract comprehensive structured information.

For semantic_description: Write a rich 200-word description covering scene content, subjects, actions, environment, mood, and thematic elements.

For structural_description: Write a 150-word analysis focusing exclusively on spatial composition, layout, object positioning, foreground/midground/background, lines of composition, and geometric structure.

For style_description: Write a 150-word analysis focusing exclusively on artistic style, color palette, lighting, texture, rendering technique, and visual treatment.

For each character detected: provide semantic (identity/appearance), emotion (expression/body language), and pose (position/orientation) descriptions of 2-3 sentences each.

For metadata: extract scene_type, time_of_day, lighting_condition, primary_subject, secondary_subjects, artistic_style, color_palette, tags, and narrative_theme.

For narrative: identify story_summary, narrative_type (cinematic/documentary/surreal/fantasy/etc.), and tone.

For emotion: identify starting_emotion, mid_emotion, end_emotion, and emotional_polarity (-1.0 to 1.0).

For objects: identify key_objects, contextual_objects, and symbolic_elements.

For low_light: score brightness, contrast, noise_estimate, shadow_dominance, and visibility_confidence (all 0.0 to 1.0)."""


def extract_image(image_input: ImageInput) -> ImageExtraction:
    """Run unified GPT-4o vision extraction on an image."""
    config = load_config()
    client = get_openai_client()

    response = client.beta.chat.completions.parse(
        model=config.models.llm_model,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Generation prompt: {image_input.generation_prompt}\n\nAnalyze this image comprehensively.",
                    },
                    image_input.to_openai_image_content(),
                ],
            },
        ],
        response_format=ImageExtraction,
        temperature=config.extraction.temperature,
        max_tokens=config.extraction.max_tokens,
    )

    parsed = response.choices[0].message.parsed
    if parsed is None:
        msg = "GPT-4o returned no parsed extraction"
        raise ValueError(msg)
    return parsed
```

Discrepancy references:
* DD-02: Requirements specify separate extraction modules (narrative.py, emotion.py, etc.) but research recommends single unified call. Plan uses unified extractor with sub-modules re-exporting parsed fields.

Success criteria:
* Single GPT-4o call extracts all dimensions
* Returns validated `ImageExtraction` Pydantic model
* Handles images via URL or base64

Context references:
* multi-vector-encoding-research.md (Lines 440-530) - Unified extraction architecture
* azure-ai-foundry-sdk-research.md (Lines 300-430) - Vision API patterns

Dependencies:
* Steps 2.1-2.3, 3.1 complete

### Step 3.4: Create extraction sub-modules (narrative, emotion, objects, low_light)

Thin wrapper modules that re-export relevant fields from the unified extraction, maintaining the module structure from requirements.

Files:
* `src/ai_search/extraction/narrative.py` - Narrative intent accessor
* `src/ai_search/extraction/emotion.py` - Emotional trajectory accessor
* `src/ai_search/extraction/objects.py` - Required objects accessor
* `src/ai_search/extraction/low_light.py` - Low-light metrics accessor

Each module provides a typed accessor function:
```python
# Example: src/ai_search/extraction/narrative.py
"""Narrative intent extraction from unified extraction output."""

from __future__ import annotations

from ai_search.models import ImageExtraction, NarrativeIntent


def get_narrative(extraction: ImageExtraction) -> NarrativeIntent:
    """Extract narrative intent from a completed extraction."""
    return extraction.narrative
```

Pattern repeats for emotion, objects, and low_light sub-modules.

Success criteria:
* Each sub-module provides typed access to its extraction dimension
* Module structure matches requirements.md Section 11

Context references:
* requirements.md Sections 4.1.1-4.1.5 - Extracted dimensions

Dependencies:
* Step 3.3 complete

## Implementation Phase 4: Embedding Generation

<!-- parallelizable: true -->

### Step 4.1: Create embeddings/encoder.py — base embedding service with Matryoshka dimensions

Core embedding service that wraps `text-embedding-3-large` with configurable dimensions.

Files:
* `src/ai_search/embeddings/encoder.py` - Base embedding encoder

Implementation:
```python
"""Base embedding encoder using text-embedding-3-large with Matryoshka dimensions."""

from __future__ import annotations

import asyncio

from openai import AsyncAzureOpenAI

from ai_search.clients import get_async_openai_client
from ai_search.config import load_config


async def embed_texts(
    texts: list[str],
    dimensions: int,
    client: AsyncAzureOpenAI | None = None,
) -> list[list[float]]:
    """Embed a batch of texts at the specified dimensionality."""
    if not texts:
        return []

    config = load_config()
    _client = client or get_async_openai_client()
    chunk_size = config.batch.embedding_chunk_size
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), chunk_size):
        chunk = texts[i : i + chunk_size]
        response = await _client.embeddings.create(
            model=config.models.embedding_model,
            input=chunk,
            dimensions=dimensions,
        )
        all_embeddings.extend([item.embedding for item in response.data])

    return all_embeddings


async def embed_text(
    text: str,
    dimensions: int,
    client: AsyncAzureOpenAI | None = None,
) -> list[float]:
    """Embed a single text at the specified dimensionality."""
    results = await embed_texts([text], dimensions, client)
    return results[0]


def embed_text_sync(text: str, dimensions: int) -> list[float]:
    """Synchronous wrapper for single text embedding."""
    return asyncio.run(embed_text(text, dimensions))
```

Success criteria:
* Supports configurable dimensions (3072, 1024, 512, 256)
* Handles batch chunking for API limits (2048 per call)
* Both async and sync interfaces

Context references:
* azure-ai-foundry-sdk-research.md (Lines 180-240) - Embedding API with dimensions
* multi-vector-encoding-research.md (Lines 80-140) - Matryoshka dimension strategy

Dependencies:
* Phase 2 complete

### Step 4.2: Create embeddings/semantic.py, structural.py, style.py — typed wrappers

Typed wrapper functions for each vector type with correct dimension defaults.

Files:
* `src/ai_search/embeddings/semantic.py` - 3072-dim semantic embeddings
* `src/ai_search/embeddings/structural.py` - 1024-dim structural embeddings
* `src/ai_search/embeddings/style.py` - 512-dim style embeddings

Example (`semantic.py`):
```python
"""Semantic embedding generation (3072 dimensions)."""

from __future__ import annotations

from ai_search.config import load_config
from ai_search.embeddings.encoder import embed_text


async def generate_semantic_vector(description: str) -> list[float]:
    """Generate a semantic embedding from a rich description."""
    config = load_config()
    return await embed_text(description, dimensions=config.index.vector_dimensions.semantic)
```

Structural and style follow the same pattern with their respective dimension configs.

Success criteria:
* Each wrapper uses the correct dimension from config
* Typed return values for mypy

Context references:
* multi-vector-encoding-research.md (Lines 20-50) - Semantic vector strategy
* multi-vector-encoding-research.md (Lines 140-200) - Structural vector strategy
* multi-vector-encoding-research.md (Lines 230-280) - Style vector strategy

Dependencies:
* Step 4.1 complete

### Step 4.3: Create embeddings/character.py — per-character sub-vector generation

Generate semantic, emotion, and pose vectors for each detected character.

Files:
* `src/ai_search/embeddings/character.py` - Character sub-vector generation

Implementation:
```python
"""Character sub-vector embedding generation."""

from __future__ import annotations

from ai_search.config import load_config
from ai_search.embeddings.encoder import embed_texts
from ai_search.models import CharacterDescription, CharacterVectors


async def generate_character_vectors(
    characters: list[CharacterDescription],
    max_slots: int | None = None,
) -> list[CharacterVectors]:
    """Generate embedding vectors for each character (capped at max_slots)."""
    config = load_config()
    slots = max_slots or config.index.max_character_slots
    chars = characters[:slots]

    if not chars:
        return []

    dims = config.index.vector_dimensions

    # Group by dimension for batch efficiency
    semantic_texts = [c.semantic for c in chars]
    emotion_texts = [c.emotion for c in chars]
    pose_texts = [c.pose for c in chars]

    semantic_vecs = await embed_texts(semantic_texts, dimensions=dims.character_semantic)
    emotion_vecs = await embed_texts(emotion_texts, dimensions=dims.character_emotion)
    pose_vecs = await embed_texts(pose_texts, dimensions=dims.character_pose)

    return [
        CharacterVectors(
            character_id=chars[i].character_id,
            semantic_vector=semantic_vecs[i],
            emotion_vector=emotion_vecs[i],
            pose_vector=pose_vecs[i],
        )
        for i in range(len(chars))
    ]
```

Success criteria:
* Caps characters at 3 slots (configurable)
* Groups embeddings by dimension for batch API efficiency
* Returns typed `CharacterVectors` models

Context references:
* multi-vector-encoding-research.md (Lines 300-420) - Character vector architecture

Dependencies:
* Step 4.1 complete

### Step 4.4: Create embeddings/pipeline.py — orchestrator that groups by dimension and parallelizes

Orchestrator that takes an `ImageExtraction` and produces all vectors in parallel, grouped by dimension.

Files:
* `src/ai_search/embeddings/pipeline.py` - Embedding pipeline orchestrator

Implementation:
```python
"""Embedding pipeline orchestrator — generates all vectors for an image."""

from __future__ import annotations

import asyncio

from ai_search.embeddings.character import generate_character_vectors
from ai_search.embeddings.semantic import generate_semantic_vector
from ai_search.embeddings.structural import generate_structural_vector
from ai_search.embeddings.style import generate_style_vector
from ai_search.models import ImageExtraction, ImageVectors


async def generate_all_vectors(extraction: ImageExtraction) -> ImageVectors:
    """Generate all embedding vectors for an extracted image in parallel."""
    semantic_task = generate_semantic_vector(extraction.semantic_description)
    structural_task = generate_structural_vector(extraction.structural_description)
    style_task = generate_style_vector(extraction.style_description)
    character_task = generate_character_vectors(extraction.characters)

    semantic_vec, structural_vec, style_vec, char_vecs = await asyncio.gather(
        semantic_task, structural_task, style_task, character_task
    )

    return ImageVectors(
        semantic_vector=semantic_vec,
        structural_vector=structural_vec,
        style_vector=style_vec,
        character_vectors=char_vecs,
    )
```

Success criteria:
* All 4 embedding calls run in parallel via `asyncio.gather`
* Returns typed `ImageVectors` model
* Handles images with 0-3 characters

Context references:
* multi-vector-encoding-research.md (Lines 440-530) - Unified pipeline architecture

Dependencies:
* Steps 4.1-4.3 complete

## Implementation Phase 5: Azure AI Search Indexing

<!-- parallelizable: true -->

### Step 5.1: Create indexing/schema.py — full index definition with all fields + HNSW config

Define the complete Azure AI Search index schema with primitive fields, vector fields, HNSW algorithm, and semantic configuration.

Files:
* `src/ai_search/indexing/schema.py` - Index schema definition

Implementation: Build the `SearchIndex` using `azure-search-documents` SDK models. Include:
* Primitive fields: `image_id` (key), `generation_prompt` (searchable), `scene_type` (filterable), `lighting_condition` (filterable), `tags` (searchable, facetable), `emotional_polarity` (filterable, sortable), `low_light_score` (filterable), `character_count` (filterable), `narrative_type` (filterable), `metadata_json` (not searchable), `extraction_json` (not searchable)
* Primary vector fields: `semantic_vector` (3072), `structural_vector` (1024), `style_vector` (512)
* Character vector fields (3 slots × 3 types): `char_0_semantic_vector` (512), `char_0_emotion_vector` (256), `char_0_pose_vector` (256), repeat for char_1 and char_2
* HNSW config: m=4, ef_construction=400, ef_search=500, cosine similarity
* Semantic ranker config: `generation_prompt` as content field
* Function to create or update the index

Success criteria:
* Index creates successfully via `index_client.create_or_update_index()`
* All vector fields reference the shared `hnsw-cosine-profile`
* Character vector fields are top-level (not nested)

Context references:
* azure-ai-search-sdk-research.md (Lines 65-200) - Field definitions
* azure-ai-search-sdk-research.md (Lines 210-290) - HNSW configuration
* multi-vector-encoding-research.md (Lines 530-600) - Index schema JSON

Dependencies:
* Phase 2 complete

### Step 5.2: Create indexing/indexer.py — batch document upload with retry logic

Batch uploader with configurable batch size, exponential backoff retry, and progress tracking.

Files:
* `src/ai_search/indexing/indexer.py` - Document batch uploader

Implementation: Create a function `upload_documents()` that:
* Accepts a list of `SearchDocument` models
* Converts to dicts, handles empty vector fields (skip if empty list)
* Splits into batches of configurable size (default 500)
* Uses `IndexDocumentsBatch.add_upload_actions()`
* Implements exponential backoff retry on 429/503
* Logs progress via structlog

Also create `build_search_document()` that assembles a `SearchDocument` from `ImageInput`, `ImageExtraction`, and `ImageVectors`, flattening character vectors into top-level fields.

Success criteria:
* Handles 429/503 with exponential backoff
* Batch size configurable via config.yaml
* Character vectors flattened correctly to `char_N_*_vector` fields

Context references:
* azure-ai-search-sdk-research.md (Lines 490-620) - Batch indexing patterns

Dependencies:
* Steps 2.2, 5.1 complete

### Step 5.3: Create indexing/cli.py — CLI entry point for index creation and document upload

CLI for index management: create index, upload documents from a JSON file.

Files:
* `src/ai_search/indexing/cli.py` - CLI entry point

Implementation:
```python
"""CLI entry point for index management."""

from __future__ import annotations

import argparse
import sys

from ai_search.indexing.schema import create_or_update_index


def main() -> None:
    """Index management CLI."""
    parser = argparse.ArgumentParser(description="AI Search Index Management")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("create", help="Create or update the search index")

    args = parser.parse_args()

    if args.command == "create":
        index = create_or_update_index()
        print(f"Index '{index.name}' created/updated successfully")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
```

Success criteria:
* `uv run ai-search-index create` creates the index
* Exits with appropriate codes on success/failure

Dependencies:
* Steps 5.1-5.2 complete

## Implementation Phase 6: Retrieval Service

<!-- parallelizable: true -->

### Step 6.1: Create retrieval/query.py — query embedding generation (text + image queries)

Generate query embeddings for text or image queries, producing semantic, structural, and style vectors.

Files:
* `src/ai_search/retrieval/query.py` - Query embedding generation

Implementation:
* For text queries: embed the raw query as semantic; use LLM to generate structural/style descriptions then embed those.
* For image queries: run GPT-4o extraction then embed the three descriptions.
* Parallel embedding generation via asyncio.gather.

Success criteria:
* Text queries produce 3 query vectors
* Image queries produce 3 query vectors via extraction pipeline
* All embedding calls run in parallel

Context references:
* hybrid-retrieval-research.md (Lines 175-250) - Multi-vector query strategy

Dependencies:
* Phase 4 complete

### Step 6.2: Create retrieval/search.py — hybrid search with configurable weights

Execute hybrid search against Azure AI Search with weighted multi-vector queries.

Files:
* `src/ai_search/retrieval/search.py` - Hybrid search execution

Implementation:
* Build `VectorizedQuery` objects for semantic (weight × 10), structural (weight × 10), style (weight × 10)
* Pass `search_text` for BM25
* Apply OData filters when provided
* Return `select` fields needed for re-ranking
* Read weights from config.yaml

Success criteria:
* Sends single search request with 3 vector queries + BM25
* Weights derived from config.yaml × 10 mapping
* Supports optional OData filters

Context references:
* azure-ai-search-sdk-research.md (Lines 310-440) - Hybrid search patterns
* hybrid-retrieval-research.md (Lines 70-170) - RRF weight mapping

Dependencies:
* Phase 2 complete

### Step 6.3: Create retrieval/reranker.py — Stage 2 rule-based re-ranking

Application-level re-ranking using emotional alignment, narrative consistency, object overlap, and low-light compatibility.

Files:
* `src/ai_search/retrieval/reranker.py` - Rule-based re-ranking

Implementation:
* `emotional_alignment_score()`: Polarity difference + trajectory label match
* `narrative_consistency_score()`: Token overlap between narrative types
* `object_overlap_score()`: Jaccard similarity on object lists
* `low_light_compatibility_score()`: Absolute difference from query preference
* `compute_rerank_score()`: Weighted combination from config
* `rerank_candidates()`: Sort by combined score, return top-N

Success criteria:
* Re-ranking completes in < 5ms for 200 candidates
* Weights configurable via config.yaml
* Handles missing fields gracefully (neutral scores)

Context references:
* hybrid-retrieval-research.md (Lines 370-500) - Re-ranking implementation

Dependencies:
* Phase 2 complete (models, config)

### Step 6.4: Create retrieval/diversity.py — Stage 3 MMR diversity

Maximal Marginal Relevance for result diversification.

Files:
* `src/ai_search/retrieval/diversity.py` - MMR diversity

Implementation:
* Accept candidate embeddings (numpy arrays) and relevance scores
* Iteratively select documents balancing relevance and inter-document dissimilarity
* Lambda parameter configurable from config.yaml (default 0.6)
* Return top-N indices

Success criteria:
* MMR completes in < 1ms for 50 candidates
* Lambda parameter configurable
* Returns diverse result set

Context references:
* hybrid-retrieval-research.md (Lines 600-720) - MMR implementation

Dependencies:
* Phase 2 complete

### Step 6.5: Create retrieval/pipeline.py — three-stage orchestrator

Orchestrate the full retrieval pipeline: Stage 1 (Azure Search) → Stage 2 (re-rank) → Stage 3 (MMR).

Files:
* `src/ai_search/retrieval/pipeline.py` - Three-stage retrieval orchestrator

Implementation:
* Accept query text or image
* Generate query embeddings
* Execute hybrid search (Stage 1)
* Apply rule-based re-ranking (Stage 2)
* Apply MMR diversity (Stage 3)
* Return final ranked results

Success criteria:
* End-to-end query returns ranked diverse results
* Total P95 latency budget: < 300ms (excluding embedding generation when cached)

Context references:
* hybrid-retrieval-research.md (Lines 730-820) - End-to-end flow

Dependencies:
* Steps 6.1-6.4 complete

## Implementation Phase 7: CLI Entry Points

<!-- parallelizable: true -->

### Step 7.1: Create ingestion/cli.py — ingest command

CLI for processing a single image or batch of images through the full ingestion pipeline.

Files:
* `src/ai_search/ingestion/cli.py` - Ingestion CLI

Implementation:
* Accept `--image-url` or `--image-file` + `--prompt` + `--image-id`
* Run extraction → embedding → index upload
* Print summary of processed document

Success criteria:
* `uv run ai-search-ingest --image-url URL --prompt "..." --image-id id1` processes and indexes

Dependencies:
* Phases 3-5 complete

### Step 7.2: Create retrieval/cli.py — query command

CLI for executing search queries.

Files:
* `src/ai_search/retrieval/cli.py` - Query CLI

Implementation:
* Accept `--query TEXT` for text search
* Accept `--top N` for result count
* Accept `--filter ODATA` for filtering
* Print ranked results with scores

Success criteria:
* `uv run ai-search-query --query "cinematic night scene" --top 10` returns results

Dependencies:
* Phase 6 complete

## Implementation Phase 8: Tests

<!-- parallelizable: true -->

### Step 8.1: Create tests/conftest.py with shared fixtures and mocks

Shared test fixtures: mock OpenAI client, mock Search client, sample config, sample image input.

Files:
* `tests/conftest.py` - Shared test fixtures

Implementation:
* `sample_config` fixture: Creates temp config.yaml
* `mock_openai_client` fixture: Returns MagicMock
* `sample_image_input` fixture: Returns `ImageInput` with test data
* `sample_extraction` fixture: Returns `ImageExtraction` with test data
* `sample_vectors` fixture: Returns `ImageVectors` with random float arrays

Success criteria:
* Fixtures usable across all test modules
* No real API calls in unit tests

Context references:
* uv-python-project-research.md (Lines 500-540) - conftest.py pattern

Dependencies:
* Phase 2 complete

### Step 8.2: Create unit tests for config, models, and client factories

Files:
* `tests/test_config.py` - Config loading tests
* `tests/test_models.py` - Model validation tests

Test scenarios:
* Config loads from YAML with correct defaults
* Config handles missing file gracefully
* All Pydantic models validate correct data
* Models reject invalid data (wrong types, out-of-range values)
* ImageExtraction works as response_format schema

Success criteria:
* 100% coverage on config.py and models.py

Dependencies:
* Phase 2 complete

### Step 8.3: Create unit tests for extraction and embedding modules

Files:
* `tests/test_extraction/test_extractor.py` - Extraction tests
* `tests/test_embeddings/test_encoder.py` - Embedding encoder tests
* `tests/test_embeddings/test_pipeline.py` - Embedding pipeline tests

Test scenarios:
* Extraction calls GPT-4o with correct message format (mocked)
* Extraction returns validated ImageExtraction model
* Encoder batches texts correctly by chunk size
* Encoder passes correct dimensions parameter
* Pipeline orchestrates 4 parallel embedding calls

Success criteria:
* All GPT-4o and embedding calls are mocked
* Verify correct parameters passed to Azure APIs

Dependencies:
* Phases 3-4 complete

### Step 8.4: Create unit tests for indexing schema and retrieval pipeline

Files:
* `tests/test_indexing/test_schema.py` - Index schema tests
* `tests/test_indexing/test_indexer.py` - Batch upload tests
* `tests/test_retrieval/test_search.py` - Search tests
* `tests/test_retrieval/test_reranker.py` - Re-ranking tests
* `tests/test_retrieval/test_diversity.py` - MMR tests

Test scenarios:
* Schema generates correct field definitions and HNSW config
* Batch upload handles retry on 429
* Search builds correct VectorizedQuery with weights
* Re-ranker computes scores correctly for each scoring function
* MMR produces diverse results (no identical nearest neighbors in top-K)

Success criteria:
* Schema field count matches expected
* Re-ranking scores are deterministic given inputs
* MMR lambda=0 returns most diverse, lambda=1 returns most relevant

Dependencies:
* Phases 5-6 complete

### Step 8.5: Create integration test markers and fixtures

Files:
* `tests/test_integration/__init__.py`
* `tests/test_integration/conftest.py` - Integration-specific fixtures
* `tests/test_integration/test_end_to_end.py` - Placeholder for E2E tests

All integration tests marked with `@pytest.mark.integration` for selective execution. These tests require real Azure credentials and are skipped by default.

Success criteria:
* `pytest -m "not integration"` runs all unit tests
* Integration tests are discoverable but skip without credentials

Dependencies:
* All prior phases complete

## Implementation Phase 9: Validation

<!-- parallelizable: false -->

### Step 9.1: Run full project validation

Execute all validation commands for the project:
* `uv run ruff check src/ tests/`
* `uv run mypy src/`
* `uv run pytest tests/ -m "not integration"`

### Step 9.2: Fix minor validation issues

Iterate on lint errors, build warnings, and test failures. Apply fixes directly when corrections are straightforward and isolated.

### Step 9.3: Report blocking issues

When validation failures require changes beyond minor fixes:
* Document the issues and affected files.
* Provide the user with next steps.
* Recommend additional research and planning rather than inline fixes.
* Avoid large-scale refactoring within this phase.

## Dependencies

* Python >= 3.11, UV, Azure AI Foundry endpoint + API key, Azure AI Search endpoint + API key

## Success Criteria

* All unit tests pass with zero failures
* Ruff reports zero lint errors
* Mypy reports zero type errors
* Package imports successfully after `uv sync`
