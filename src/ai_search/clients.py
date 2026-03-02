"""Client factories for Azure AI Foundry and Azure AI Search."""

from __future__ import annotations

from functools import lru_cache

import httpx
import structlog
from azure.ai.inference.aio import EmbeddingsClient, ImageEmbeddingsClient
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from openai import AsyncAzureOpenAI, AzureOpenAI

from ai_search.config import (
    load_blob_secrets,
    load_cv_secrets,
    load_foundry_secrets,
    load_openai_secrets,
    load_search_secrets,
)

logger = structlog.get_logger(__name__)

_AZURE_COGNITIVE_SCOPE = "https://cognitiveservices.azure.com/.default"


@lru_cache(maxsize=1)
def _get_credential() -> DefaultAzureCredential:
    """Return a cached DefaultAzureCredential."""
    return DefaultAzureCredential()


@lru_cache(maxsize=1)
def _get_token_provider() -> object:
    """Return a cached token provider for Azure OpenAI."""
    return get_bearer_token_provider(_get_credential(), _AZURE_COGNITIVE_SCOPE)


@lru_cache(maxsize=1)
def get_openai_client() -> AzureOpenAI:
    """Return a cached synchronous Azure OpenAI client using Entra ID auth."""
    secrets = load_foundry_secrets()
    api = load_openai_secrets()
    return AzureOpenAI(
        azure_endpoint=secrets.endpoint,
        azure_ad_token_provider=_get_token_provider(),
        api_version=api.api_version,
    )


@lru_cache(maxsize=1)
def get_async_openai_client() -> AsyncAzureOpenAI:
    """Return a cached asynchronous Azure OpenAI client using Entra ID auth."""
    secrets = load_foundry_secrets()
    api = load_openai_secrets()
    return AsyncAzureOpenAI(
        azure_endpoint=secrets.endpoint,
        azure_ad_token_provider=_get_token_provider(),
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


@lru_cache(maxsize=1)
def get_foundry_embed_client() -> EmbeddingsClient:
    """Return a cached Azure AI Inference EmbeddingsClient.

    Uses the Azure AI Inference SDK with ``DefaultAzureCredential`` for
    Entra ID authentication against the Foundry models endpoint.
    """
    secrets = load_foundry_secrets()
    if not secrets.embed_endpoint:
        msg = (
            "AZURE_FOUNDRY_EMBED_ENDPOINT is not configured. "
            "Set it to the Foundry models endpoint, e.g. "
            "https://<resource>.services.ai.azure.com/models"
        )
        raise ValueError(msg)
    return EmbeddingsClient(
        endpoint=secrets.embed_endpoint,
        credential=_get_credential(),
        credential_scopes=[_AZURE_COGNITIVE_SCOPE],
    )


@lru_cache(maxsize=1)
def get_foundry_image_embed_client() -> ImageEmbeddingsClient:
    """Return a cached Azure AI Inference ImageEmbeddingsClient.

    Uses the Azure AI Inference SDK with ``DefaultAzureCredential`` for
    Entra ID authentication. Routes to ``POST /images/embeddings`` for
    visual image embedding (as opposed to text tokenization).
    """
    secrets = load_foundry_secrets()
    if not secrets.embed_endpoint:
        msg = (
            "AZURE_FOUNDRY_EMBED_ENDPOINT is not configured. "
            "Set it to the Foundry models endpoint, e.g. "
            "https://<resource>.services.ai.azure.com/models"
        )
        raise ValueError(msg)
    return ImageEmbeddingsClient(
        endpoint=secrets.embed_endpoint,
        credential=_get_credential(),
        credential_scopes=[_AZURE_COGNITIVE_SCOPE],
    )


@lru_cache(maxsize=1)
def get_cv_client() -> httpx.AsyncClient:
    """Return a cached async HTTP client for Azure Computer Vision."""
    secrets = load_cv_secrets()
    if not secrets.endpoint or not secrets.api_key:
        logger.error(
            "Azure Computer Vision secrets not configured",
            endpoint_set=secrets.endpoint is not None,
            api_key_set=secrets.api_key is not None,
        )
        msg = (
            "Azure Computer Vision secrets not configured. "
            "Set AZURE_CV_ENDPOINT and AZURE_CV_API_KEY environment variables."
        )
        raise ValueError(msg)
    return httpx.AsyncClient(
        base_url=secrets.endpoint,
        headers={"Ocp-Apim-Subscription-Key": secrets.api_key},
        timeout=30.0,
    )


@lru_cache(maxsize=1)
def get_blob_container_client():
    """Return a cached Azure Blob Storage container client.

    Uses DefaultAzureCredential (Entra ID) authentication.
    Returns None if storage is not configured.
    """
    from azure.storage.blob import ContainerClient

    secrets = load_blob_secrets()
    if not secrets.account_url:
        logger.warning("Azure Blob Storage not configured, skipping cloud upload")
        return None
    return ContainerClient(
        account_url=secrets.account_url,
        container_name=secrets.container_name,
        credential=_get_credential(),
    )
