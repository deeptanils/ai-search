<!-- markdownlint-disable-file -->
# Task Research: Architecture & README Documentation Update

Update architecture.md with granular API details and README.md with comprehensive setup/run instructions.

## Task Implementation Requests

* Update `docs/architecture.md` with granular API information (SDK classes, endpoints, routes, auth methods, config details)
* Update `README.md` with installation, configuration, ingestion, and search instructions

## Scope and Success Criteria

* Scope: Both documentation files, covering all Azure services, SDK packages, API calls, authentication, configuration, CLI usage, and development workflow
* Assumptions: Basic tier is the current deployment; Foundry backend is the default for image embeddings
* Success Criteria:
  * architecture.md contains all API calls with SDK methods, routes, models, and locations
  * architecture.md includes full index schema with field attributes, HNSW config, scoring profiles
  * architecture.md documents all environment variables, config models, and data models
  * README.md provides step-by-step installation, configuration, and usage instructions
  * README.md covers batch ingestion, single image ingestion, and search commands

## Research Executed

### Subagent Research

* Subagent output: `.copilot-tracking/research/subagents/2026-02-27/architecture-details-research.md`
  * 11 sections covering: API endpoints, SDK packages, config, embedding pipeline, index schema, ingestion, retrieval, project structure, scripts, data models, GPT-4o extraction, implementation patterns

### Files Analyzed

* `pyproject.toml` — dependencies, entry points, tool config
* `config.yaml` — full configuration
* `.env.example` — environment variable template
* `data/sample_images.json` — 10 sample images
* `docs/architecture.md` — existing architecture document (250 lines)
* `README.md` — existing README (100 lines, outdated)

## Key Discoveries

* README referenced `uv run` commands which don't work with this project setup (requires `source .venv/bin/activate`)
* README referenced S2 Standard tier — project has been simplified to Basic tier
* README mentioned Florence as the primary image backend — Foundry (embed-v-4-0) is now the default
* Architecture doc was missing: API call details, SDK classes, auth methods, HNSW config, scoring profiles, relevance scoring, GPT-4o extraction details, CLI arguments

## Changes Made

### architecture.md

* Added "Azure Services and SDK Dependencies" section with service table, SDK packages table, authentication methods table
* Added "API Calls Reference" section covering all OpenAI, AI Inference, Computer Vision, and Search SDK calls with methods, models, routes, and locations
* Expanded "Index Schema" with full field attribute table (SDK class, filterable, sortable, facetable, searchable), HNSW algorithm configuration code, scoring profile code
* Added "GPT-4o Extraction Pipeline" section with API call details, extraction output model, metadata generation
* Expanded "Embedding Pipeline" with orchestration details, text/image embedding tables, preprocessing specs, vector validation
* Added "Ingestion Pipeline" with single image flow, batch ingestion parameters, retry logic, sample data
* Expanded "Retrieval Pipeline" with query vector generation, hybrid search execution, weight scaling, relevance scoring metrics and confidence tiers
* Added full "Configuration" section with config.yaml contents, Pydantic model hierarchy, environment variables table
* Added "Data Models" summary table
* Added "CLI Entry Points" with argument tables for all 3 commands
* Retained "Future Implementation" section unchanged

### README.md

* Rewrote "Prerequisites" with correct requirements (Python 3.11+, UV, Azure resources)
* Added step-by-step "Installation" with venv activation instructions
* Added macOS SSL certificate fix note
* Rewrote "Configuration" with full `.env` template, variable table, auth explanation, and config.yaml contents
* Added structured "Usage" section with Step 1 (create index), Step 2 (ingest images), Step 3 (search)
* Step 2 covers both batch ingestion (with flags) and single image ingestion (URL and file)
* Added "What Happens During Ingestion" explanation
* Added search scripts reference
* Rewrote "Development" with correct test commands, linting, project structure
* Removed outdated references to S2 tier, Florence-first backend, `uv run` commands
