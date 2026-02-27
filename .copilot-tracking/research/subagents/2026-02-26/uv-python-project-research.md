# UV Python Project Setup — Research Document

> **Subagent**: researcher-subagent
> **Date**: 2026-02-26
> **Workspace**: `/Users/deesaha/Desktop/WORK/projects/ai-search`
> **Status**: Complete

---

## Executive Summary

This document contains deep research on setting up a production-grade Python project using **UV** (the fast Python package manager by Astral) for the Candidate Generation & AI Search Pipeline. The project requires Azure AI Foundry for model hosting and Azure AI Search for vector retrieval. All findings below are written as actionable guidance for implementation.

---

## 1. UV Basics

### 1.1 What is UV?

UV is a drop-in replacement for pip, pip-tools, virtualenv, and parts of Poetry/PDM. Written in Rust, it is 10-100x faster than pip. It handles:

- Virtual environment creation
- Dependency resolution and installation
- Lock file generation
- Project initialization
- Script execution within managed environments

### 1.2 Initializing a Project

```bash
# Initialize a new project in the current directory
uv init

# Initialize with a specific name
uv init ai-search

# Initialize as a library (vs application)
uv init --lib ai-search

# Initialize as an application (default)
uv init --app ai-search
```

**For this project, use `--app`** since this is a pipeline application, not a reusable library.

Running `uv init` creates:

- `pyproject.toml` — project metadata and dependencies
- `.python-version` — pinned Python version
- `README.md` — placeholder readme
- `hello.py` or `src/<name>/__init__.py` — depending on `--app` or `--lib`

### 1.3 Virtual Environment Management

UV automatically creates and manages a `.venv` directory:

```bash
# UV creates .venv automatically on first `uv sync` or `uv run`
uv sync

# Explicitly create a venv
uv venv

# Create venv with specific Python version
uv venv --python 3.12

# Activate manually (rarely needed — uv run handles this)
source .venv/bin/activate
```

**Key insight**: With UV, you rarely need to activate the virtual environment. `uv run` automatically executes commands within the project's virtualenv.

### 1.4 Python Version Management

UV can install and manage Python versions:

```bash
# Install a specific Python version
uv python install 3.12

# Pin the project to a version
uv python pin 3.12
```

This creates/updates `.python-version` in the project root.

---

## 2. pyproject.toml Configuration

### 2.1 Recommended pyproject.toml for This Project

```toml
[project]
name = "ai-search"
version = "0.1.0"
description = "Candidate Generation & AI Search Pipeline using Azure AI Foundry and Azure AI Search"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
authors = [
    { name = "Your Name", email = "you@example.com" }
]

dependencies = [
    "openai>=1.0",
    "azure-search-documents>=11.4.0",
    "azure-identity>=1.15.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "pyyaml>=6.0",
    "python-dotenv>=1.0",
    "pillow>=10.0",
    "httpx>=0.27",
    "structlog>=24.0",
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
markers = [
    "integration: marks tests as integration tests (deselect with '-m \"not integration\"')",
    "slow: marks tests as slow",
]
```

### 2.2 Key Design Decisions

#### Build backend: Hatchling

UV does not provide its own build backend. The recommended backends are:

| Backend | When to use |
|---|---|
| `hatchling` | Default for UV projects. Modern, fast, flexible. |
| `setuptools` | Legacy projects or specific needs. |
| `flit-core` | Simple pure-Python packages. |

**Recommendation**: Use `hatchling` — it's what UV defaults to and supports src-layout natively.

#### dependency-groups vs optional-dependencies

UV (as of late 2025+) uses **PEP 735 dependency groups** rather than the older `[project.optional-dependencies]` pattern:

```toml
# Modern UV approach (PEP 735)
[dependency-groups]
dev = [
    "pytest>=8.0",
    "ruff>=0.5",
]

# Older approach (still works but less idiomatic with UV)
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.5",
]
```

**Recommendation**: Use `[dependency-groups]` for dev dependencies. This is the UV-native approach.

#### requires-python

Set to `>=3.11` because:

- Azure SDK libraries require 3.8+ but benefit from 3.11+ performance
- Pydantic v2 benefits from 3.11+ typing features
- `tomllib` is stdlib in 3.11+
- `asyncio.TaskGroup` available in 3.11+

---

## 3. Project Structure

### 3.1 Recommended Directory Layout

```
ai-search/
├── src/
│   └── ai_search/            # Top-level package (underscore, not hyphen)
│       ├── __init__.py        # Package marker + version
│       ├── config.py          # Pydantic settings + YAML loader
│       ├── models.py          # Shared Pydantic models
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── cli.py         # CLI entry point
│       │   ├── loader.py      # Image loading
│       │   └── metadata.py    # LLM metadata generation
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
│       │   ├── schema.py      # Azure Search index definition
│       │   └── indexer.py     # Document upload
│       └── retrieval/
│           ├── __init__.py
│           ├── cli.py
│           ├── query.py       # Query generation
│           ├── search.py      # Azure Search client
│           └── reranker.py    # Re-ranking logic
├── tests/
│   ├── __init__.py            # OPTIONAL — pytest finds tests without it
│   ├── conftest.py            # Shared fixtures
│   ├── test_ingestion/
│   │   ├── __init__.py
│   │   └── test_metadata.py
│   ├── test_extraction/
│   ├── test_embeddings/
│   ├── test_indexing/
│   └── test_retrieval/
├── config.yaml
├── .env
├── .env.example               # Committed template (no secrets)
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

### 3.2 Key Structural Questions Answered

#### Should `src/` have an `__init__.py`?

**No.** The `src/` directory is NOT a Python package — it's a namespace directory used by the src-layout convention. The actual package root is `src/ai_search/`.

#### Should each subpackage have `__init__.py`?

**Yes.** Every subdirectory that should be importable as a Python package must have an `__init__.py`. This includes:

- `src/ai_search/__init__.py` — required (top-level package)
- `src/ai_search/ingestion/__init__.py` — required
- `src/ai_search/extraction/__init__.py` — required
- etc.

The `__init__.py` can be empty or can re-export key symbols:

```python
# src/ai_search/__init__.py
"""Candidate Generation & AI Search Pipeline."""

__version__ = "0.1.0"
```

```python
# src/ai_search/ingestion/__init__.py
"""Ingestion module — loads images and generates synthetic metadata."""
```

#### How to handle imports?

With the src-layout and `tool.hatch.build.targets.wheel.packages = ["src/ai_search"]` plus `tool.pytest.ini_options.pythonpath = ["src"]`:

```python
# Absolute imports (preferred)
from ai_search.config import Settings
from ai_search.ingestion.metadata import generate_metadata
from ai_search.embeddings.semantic import SemanticEncoder

# Within the same subpackage
from .metadata import generate_metadata  # relative import
```

**No sys.path hacks needed.** UV + hatchling + src-layout handles everything.

### 3.3 Why src-layout?

| Benefit | Explanation |
|---|---|
| Prevents accidental import of uninstalled code | Forces you to install the package before testing |
| Clean separation | Source code ≠ project root |
| Matches hatchling default | Zero additional config |
| Industry standard | Used by most modern Python projects |

---

## 4. Configuration Management

### 4.1 The Two-File Pattern

The project requirements mandate:

| File | Purpose | Committed to Git? |
|---|---|---|
| `.env` | Secrets (API keys, endpoints) | **No** (`.gitignore`) |
| `config.yaml` | Non-secret configuration (model names, weights, dimensions) | **Yes** |

### 4.2 Pydantic Settings Pattern

```python
# src/ai_search/config.py
from __future__ import annotations

import yaml
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AzureFoundrySettings(BaseSettings):
    """Secrets loaded from .env file."""
    model_config = SettingsConfigDict(
        env_prefix="AZURE_FOUNDRY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    endpoint: str
    api_key: str


class AzureSearchSettings(BaseSettings):
    """Secrets loaded from .env file."""
    model_config = SettingsConfigDict(
        env_prefix="AZURE_AI_SEARCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    endpoint: str
    api_key: str
    index_name: str = "candidate-index"


class ModelConfig:
    """Non-secret config loaded from config.yaml."""
    embedding_model: str
    llm_model: str


class SearchWeights:
    semantic_weight: float
    structural_weight: float
    style_weight: float
    keyword_weight: float


class AppConfig:
    """Combined configuration from .env and config.yaml."""

    def __init__(self, config_path: Path = Path("config.yaml")) -> None:
        # Load secrets from .env
        self.foundry = AzureFoundrySettings()
        self.search = AzureSearchSettings()

        # Load non-secret config from YAML
        with open(config_path) as f:
            yaml_data = yaml.safe_load(f)

        self.models = yaml_data.get("models", {})
        self.search_weights = yaml_data.get("search", {})
        self.index_config = yaml_data.get("index", {})
```

### 4.3 Why This Pattern?

1. **pydantic-settings** automatically reads `.env` files AND environment variables (env vars override `.env`)
2. **Separation of concerns**: Secrets in `.env`, structure in YAML
3. **Validation**: Pydantic validates types, required fields, constraints
4. **12-factor compliance**: Environment variables for secrets, config files for structure
5. **Testable**: Easy to override settings in tests

### 4.4 .env.example Template

Commit this file (with placeholder values) so new developers know what's needed:

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

---

## 5. UV Lock File

### 5.1 How `uv.lock` Works

- Generated automatically when you run `uv sync` or `uv lock`
- Contains exact resolved versions of ALL dependencies (direct + transitive)
- Platform-aware — includes markers for OS/Python version
- Uses a custom format (not `requirements.txt`)
- **Must be committed to Git** for reproducible builds

### 5.2 Key Commands

```bash
# Generate/update the lock file
uv lock

# Install exactly what's in the lock file
uv sync

# Update a specific package in the lock
uv lock --upgrade-package openai

# Update all packages
uv lock --upgrade

# Install including dev dependencies (default)
uv sync

# Install without dev dependencies (production)
uv sync --no-dev

# Install only a specific dependency group
uv sync --group dev
```

### 5.3 Lock File in CI/CD

```bash
# In CI, use --frozen to fail if lock is out of date
uv sync --frozen
```

This ensures the lock file matches `pyproject.toml` — if someone added a dependency but forgot to update the lock, CI fails.

---

## 6. UV Commands Reference

### 6.1 Core Workflow Commands

| Command | Purpose |
|---|---|
| `uv init` | Initialize a new project |
| `uv add <pkg>` | Add a dependency to `pyproject.toml` and install it |
| `uv remove <pkg>` | Remove a dependency |
| `uv sync` | Install all dependencies from lock file |
| `uv lock` | Resolve and write the lock file |
| `uv run <cmd>` | Run a command in the project's virtualenv |

### 6.2 Adding Dependencies

```bash
# Add a runtime dependency
uv add openai
uv add "azure-search-documents>=11.4"
uv add pydantic pydantic-settings pyyaml python-dotenv

# Add to a dependency group
uv add --group dev pytest ruff mypy

# Add with version constraints
uv add "pillow>=10.0,<12.0"

# Add from a git repo
uv add "git+https://github.com/user/repo.git"
```

### 6.3 Running Commands

```bash
# Run a Python script
uv run python -m ai_search.ingestion.cli

# Run pytest
uv run pytest

# Run ruff linting
uv run ruff check src/

# Run mypy type checking
uv run mypy src/

# Run a defined script entry point
uv run ai-search-ingest
```

### 6.4 Low-Level pip Interface

UV also provides a pip-compatible interface (useful for Docker/legacy):

```bash
# Install from requirements.txt
uv pip install -r requirements.txt

# Install a single package
uv pip install openai

# Compile requirements (like pip-tools)
uv pip compile pyproject.toml -o requirements.txt
```

### 6.5 Environment Inspection

```bash
# Show installed packages
uv pip list

# Show dependency tree
uv tree

# Show project info
uv version
```

---

## 7. Testing Setup

### 7.1 pytest Configuration in pyproject.toml

Already included in Section 2.1 above. Key settings:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]           # Critical for src-layout imports
addopts = "-v --tb=short --strict-markers"
```

The `pythonpath = ["src"]` setting tells pytest to add `src/` to `sys.path`, enabling `from ai_search.xxx import yyy` in test files.

### 7.2 Test Structure

```
tests/
├── conftest.py                # Shared fixtures
├── test_ingestion/
│   ├── conftest.py            # Ingestion-specific fixtures
│   ├── test_metadata.py
│   └── test_loader.py
├── test_extraction/
│   ├── test_narrative.py
│   └── test_emotion.py
├── test_embeddings/
│   └── test_semantic.py
├── test_indexing/
│   └── test_schema.py
└── test_retrieval/
    ├── test_query.py
    └── test_reranker.py
```

### 7.3 conftest.py Pattern

```python
# tests/conftest.py
import pytest
from pathlib import Path
from unittest.mock import MagicMock


@pytest.fixture
def sample_config_path(tmp_path: Path) -> Path:
    """Create a temporary config.yaml for testing."""
    config = tmp_path / "config.yaml"
    config.write_text("""
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
""")
    return config


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for unit tests."""
    return MagicMock()
```

### 7.4 Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=ai_search --cov-report=term-missing

# Run specific module
uv run pytest tests/test_ingestion/

# Skip integration tests
uv run pytest -m "not integration"

# Run in parallel (requires pytest-xdist)
uv run pytest -n auto
```

---

## 8. Complete Bootstrap Sequence

Here is the exact sequence of commands to bootstrap this project:

```bash
# 1. Initialize the project
cd /Users/deesaha/Desktop/WORK/projects/ai-search
uv init --app --name ai-search

# 2. Pin Python version
uv python pin 3.12

# 3. Add runtime dependencies
uv add openai "azure-search-documents>=11.4" "azure-identity>=1.15" \
    "pydantic>=2.0" "pydantic-settings>=2.0" "pyyaml>=6.0" \
    "python-dotenv>=1.0" "pillow>=10.0" "httpx>=0.27" "structlog>=24.0"

# 4. Add dev dependencies
uv add --group dev "pytest>=8.0" "pytest-asyncio>=0.23" "pytest-cov>=5.0" \
    "ruff>=0.5" "mypy>=1.10" "types-PyYAML>=6.0" "types-Pillow>=10.0"

# 5. Create project structure
mkdir -p src/ai_search/{ingestion,extraction,embeddings,indexing,retrieval}
mkdir -p tests/{test_ingestion,test_extraction,test_embeddings,test_indexing,test_retrieval}

# 6. Create __init__.py files
touch src/ai_search/__init__.py
touch src/ai_search/{ingestion,extraction,embeddings,indexing,retrieval}/__init__.py
touch tests/conftest.py

# 7. Sync everything
uv sync

# 8. Verify
uv run python -c "import ai_search; print('OK')"
uv run pytest --co  # collect tests (dry run)
```

---

## 9. Workspace Findings

### 9.1 Current State

| Item | Status |
|---|---|
| `pyproject.toml` | **Does not exist** — must be created |
| `config.yaml` | **Does not exist** — must be created |
| `.env` | **Does not exist** — must be created |
| `src/` directory | **Does not exist** — must be created |
| `tests/` directory | **Does not exist** — must be created |
| `requirements.md` | **Exists** — comprehensive requirements (v1 + v2) |
| UV installed | **Unknown** — must verify with `which uv` |

### 9.2 Requirements Analysis

From the requirements document, the project needs:

- **Azure AI Foundry** for all LLM and embedding calls (via `openai` SDK with Azure config)
- **Azure AI Search** for vector indexing and hybrid retrieval
- **Five pipeline stages**: ingestion, extraction, embeddings, indexing, retrieval
- **Configuration split**: `.env` for secrets, `config.yaml` for non-secrets
- **Dependency manager**: UV (explicitly stated in Section 10)

---

## 10. Recommended Next Research Topics

1. **Azure AI Foundry SDK patterns** — How to use the `openai` Python SDK configured for Azure endpoints, including structured outputs for metadata generation.

2. **Azure AI Search Python SDK** — Index creation with multiple vector fields, HNSW configuration, hybrid search queries, scoring profiles.

3. **Multi-vector embedding architecture** — How to generate semantic, structural, and style embeddings using Azure-hosted models. Fallback strategies when SigLIP/DINOv2 aren't available on Azure Foundry.

4. **Pydantic v2 data models** — Designing the document schema, extraction output models, and search result models with full validation.

5. **Async pipeline design** — Whether to use `asyncio` for parallel embedding generation and search queries; `httpx` async client patterns.

6. **Docker/containerization with UV** — How to build a production Docker image with UV for deployment.

7. **CI/CD pipeline** — GitHub Actions workflow using UV for testing, linting, and type checking.

---

## Appendix A: .gitignore Template

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

# UV
# Do NOT ignore uv.lock — it must be committed
```

## Appendix B: Key Package Versions (as of 2026-02)

| Package | Recommended Version | Purpose |
|---|---|---|
| `openai` | `>=1.0` | Azure AI Foundry access via OpenAI SDK |
| `azure-search-documents` | `>=11.4` | Azure AI Search client |
| `azure-identity` | `>=1.15` | Azure authentication (DefaultAzureCredential) |
| `pydantic` | `>=2.0` | Data validation and models |
| `pydantic-settings` | `>=2.0` | Settings from .env / env vars |
| `pyyaml` | `>=6.0` | YAML config parsing |
| `python-dotenv` | `>=1.0` | .env file loading |
| `pillow` | `>=10.0` | Image processing |
| `httpx` | `>=0.27` | Async HTTP client |
| `structlog` | `>=24.0` | Structured logging |
