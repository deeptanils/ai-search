"""Configuration management — loads .env secrets and config.yaml settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel
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
    embed_endpoint: str | None = None
    api_key: str | None = None


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


class AzureBlobStorageSecrets(BaseSettings):
    """Azure Blob Storage secrets from .env."""

    model_config = SettingsConfigDict(
        env_prefix="AZURE_STORAGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    account_name: str | None = None
    container_name: str = "images"

    @property
    def account_url(self) -> str | None:
        """Derive the blob endpoint URL from the account name."""
        if not self.account_name:
            return None
        return f"https://{self.account_name}.blob.core.windows.net"


class AzureComputerVisionSecrets(BaseSettings):
    """Azure Computer Vision (Florence) secrets from .env."""

    model_config = SettingsConfigDict(
        env_prefix="AZURE_CV_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    endpoint: str | None = None
    api_key: str | None = None
    api_version: str = "2024-02-01"
    model_version: str = "2023-04-15"


class ModelsConfig(BaseModel):
    """Model deployment names."""

    embedding_model: str = "text-embedding-3-large"
    llm_model: str = "gpt-4o"
    image_embedding_model: str = "embed-v-4-0"


class SearchWeightsConfig(BaseModel):
    """Retrieval weight configuration."""

    semantic_weight: float = 0.4
    structural_weight: float = 0.15
    style_weight: float = 0.15
    image_weight: float = 0.2
    keyword_weight: float = 0.1


class VectorDimensionsConfig(BaseModel):
    """Vector dimension configuration."""

    semantic: int = 3072
    structural: int = 1024
    style: int = 512
    image: int = 1024


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


class RetrievalConfig(BaseModel):
    """Retrieval pipeline configuration."""

    top_k: int = 50
    k_nearest: int = 100


class ExtractionConfig(BaseModel):
    """GPT-4o extraction configuration (ingestion-time)."""

    image_detail: str = "high"
    temperature: float = 0.2
    max_tokens: int = 5000


class QueryExtractionConfig(BaseModel):
    """GPT-4o extraction configuration (query-time image search).

    Uses ``high`` detail for richer cinematic descriptions.
    ``max_image_size`` controls resize before sending to GPT-4o.
    """

    image_detail: str = "high"
    temperature: float = 0.0
    max_tokens: int = 1200
    max_image_size: int = 768


class BatchConfig(BaseModel):
    """Batch processing configuration."""

    index_batch_size: int = 500
    embedding_chunk_size: int = 2048
    max_concurrent_requests: int = 50


class AppConfig(BaseModel):
    """Full application configuration from config.yaml."""

    models: ModelsConfig = ModelsConfig()
    search: SearchWeightsConfig = SearchWeightsConfig()
    image_search: SearchWeightsConfig = SearchWeightsConfig(
        semantic_weight=0.15,
        structural_weight=0.05,
        style_weight=0.05,
        image_weight=0.65,
        keyword_weight=0.1,
    )
    index: IndexConfig = IndexConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    extraction: ExtractionConfig = ExtractionConfig()
    query_extraction: QueryExtractionConfig = QueryExtractionConfig()
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
    return AzureOpenAISecrets()


@lru_cache(maxsize=1)
def load_search_secrets() -> AzureSearchSecrets:
    """Load Azure AI Search secrets from .env."""
    return AzureSearchSecrets()  # type: ignore[call-arg]


@lru_cache(maxsize=1)
def load_blob_secrets() -> AzureBlobStorageSecrets:
    """Load Azure Blob Storage secrets from .env."""
    return AzureBlobStorageSecrets()


@lru_cache(maxsize=1)
def load_cv_secrets() -> AzureComputerVisionSecrets:
    """Load Azure Computer Vision secrets from .env."""
    return AzureComputerVisionSecrets()
