---
title: "IV-014: AzureComputerVisionSecrets Empty String Defaults Research"
description: Research into the secrets configuration patterns, impact analysis, and recommended fix for the silent misconfiguration risk in AzureComputerVisionSecrets
author: copilot
ms.date: 2026-02-26
ms.topic: reference
keywords:
  - secrets
  - configuration
  - security
  - florence
  - computer-vision
---

## Finding Summary

Finding IV-014 (Major, Security) identifies that `AzureComputerVisionSecrets` uses `endpoint: str = ""` and `api_key: str = ""`, unlike `AzureFoundrySecrets` and `AzureSearchSecrets`, which declare no defaults and raise `ValidationError` at startup if environment variables are missing. The empty string defaults create a silent misconfiguration risk where an unconfigured Florence endpoint produces confusing HTTP errors rather than a clear missing-credentials message.

## Existing Secrets Patterns

### AzureFoundrySecrets (required service, fail-fast)

```python
class AzureFoundrySecrets(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AZURE_FOUNDRY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    endpoint: str       # No default → ValidationError if AZURE_FOUNDRY_ENDPOINT missing
    api_key: str        # No default → ValidationError if AZURE_FOUNDRY_API_KEY missing
```

Source: [src/ai_search/config.py](src/ai_search/config.py#L13-L26)

### AzureSearchSecrets (required service, fail-fast)

```python
class AzureSearchSecrets(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AZURE_AI_SEARCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    endpoint: str                       # No default → ValidationError
    api_key: str                        # No default → ValidationError
    index_name: str = "candidate-index" # Sensible default (not a credential)
```

Source: [src/ai_search/config.py](src/ai_search/config.py#L42-L55)

### AzureOpenAISecrets (optional, sensible default)

```python
class AzureOpenAISecrets(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AZURE_OPENAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_version: str = "2024-12-01-preview"  # Not a credential, safe default
```

Source: [src/ai_search/config.py](src/ai_search/config.py#L29-L39)

### AzureComputerVisionSecrets (optional service, silent fail)

```python
class AzureComputerVisionSecrets(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AZURE_CV_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    endpoint: str = ""                  # Empty default → creates client with blank base_url
    api_key: str = ""                   # Empty default → sends empty API key header
    api_version: str = "2024-02-01"     # Not a credential, safe default
```

Source: [src/ai_search/config.py](src/ai_search/config.py#L58-L70)

## Why AzureComputerVisionSecrets Differs

The other secrets classes follow a clear convention: credentials that must be present have no defaults. Pydantic raises `ValidationError` immediately when the environment variable is absent. The empty string defaults on `AzureComputerVisionSecrets` were introduced intentionally because Florence is an optional feature (only used when `image_url` or `image_bytes` is provided). Without defaults, `load_cv_secrets()` would raise even for users who never use Florence.

The problem: empty strings are ambiguous sentinels. They create an `httpx.AsyncClient` with a blank `base_url` and an empty `Ocp-Apim-Subscription-Key` header. The resulting HTTP errors are cryptic and unrelated to the actual cause (missing configuration).

## Caller Analysis

### All callers of `load_cv_secrets()`

| Caller                                | File                                                                     | Call Timing | Purpose                   |
|---------------------------------------|--------------------------------------------------------------------------|-------------|---------------------------|
| `get_cv_client()`                     | [src/ai_search/clients.py](src/ai_search/clients.py#L67-L74)            | Lazy        | Build httpx client        |
| `embed_image()`                       | [src/ai_search/embeddings/image.py](src/ai_search/embeddings/image.py#L41) | Lazy        | Get `api_version` param   |
| `embed_text_for_image_search()`       | [src/ai_search/embeddings/image.py](src/ai_search/embeddings/image.py#L78) | Lazy        | Get `api_version` param   |

No other source files call `load_cv_secrets()`.

### All callers of `get_cv_client()`

| Caller                                | File                                                                     | Call Timing |
|---------------------------------------|--------------------------------------------------------------------------|-------------|
| `embed_image()`                       | [src/ai_search/embeddings/image.py](src/ai_search/embeddings/image.py#L40) | Lazy        |
| `embed_text_for_image_search()`       | [src/ai_search/embeddings/image.py](src/ai_search/embeddings/image.py#L77) | Lazy        |

### Call chain to Florence functions

The pipeline in [src/ai_search/embeddings/pipeline.py](src/ai_search/embeddings/pipeline.py#L30-L32) only calls `embed_image()` when an image URL or bytes is provided:

```python
has_image = bool(image_url or image_bytes)
if has_image:
    tasks.append(embed_image(image_url=image_url, image_bytes=image_bytes))
```

All paths are **lazy**: secrets load on first function call, never at import time. The `@lru_cache(maxsize=1)` on `load_cv_secrets()` means the first call resolves secrets and all subsequent calls use the cache.

## Impact Analysis by Approach

### Option A: Remove defaults (match other secrets classes)

```python
endpoint: str   # No default
api_key: str    # No default
```

Pros:

* Consistent with `AzureFoundrySecrets` and `AzureSearchSecrets`.
* Immediate `ValidationError` with a clear message.

Cons:

* **Breaks users who do not configure Florence.** Even though calls are lazy, any code path that constructs `AzureComputerVisionSecrets()` (via `load_cv_secrets()`) would fail. Users who never use image embeddings would need to set dummy env variables.
* Forces Florence from optional to effectively required.
* Violates the design intent where Florence is opt-in.

Verdict: **Rejected.** Changes the service's opt-in contract.

### Option B: Add `enabled: bool = False` flag

```python
enabled: bool = False
endpoint: str = ""
api_key: str = ""
```

Pros:

* Explicit enablement control.
* Could guard all usage on `if not secrets.enabled: raise`.

Cons:

* Adds a new configuration concept not used elsewhere in the codebase.
* Requires users to set `AZURE_CV_ENABLED=true` in `.env` in addition to the endpoint and key.
* Overengineered for a two-field validation problem.
* Empty strings remain as defaults even when `enabled=True`.

Verdict: **Rejected.** Adds unnecessary complexity.

### Option C: Keep defaults, validate in `get_cv_client()`

```python
# Secrets stay the same
endpoint: str = ""
api_key: str = ""

# Validation moves to the client factory
def get_cv_client() -> httpx.AsyncClient:
    secrets = load_cv_secrets()
    if not secrets.endpoint or not secrets.api_key:
        raise ValueError("Azure CV endpoint and api_key must be set...")
    ...
```

Pros:

* Lazy validation at the point of use.
* Preserves opt-in nature.
* Clear error message.

Cons:

* Empty strings remain valid at the model level, which is semantically wrong.
* `embed_image()` and `embed_text_for_image_search()` also call `load_cv_secrets()` directly (for `api_version`), so the guard in `get_cv_client()` doesn't cover all paths.
* Validation logic is outside the Pydantic model, duplicating concerns.

Verdict: **Viable but not ideal.** Leaves the model semantically incorrect.

### Option D: Use `Optional[str] = None` with validation at usage

```python
endpoint: str | None = None
api_key: str | None = None
api_version: str = "2024-02-01"
```

Combine with a validation guard in `get_cv_client()`:

```python
def get_cv_client() -> httpx.AsyncClient:
    secrets = load_cv_secrets()
    if not secrets.endpoint or not secrets.api_key:
        msg = (
            "Azure Computer Vision is not configured. "
            "Set AZURE_CV_ENDPOINT and AZURE_CV_API_KEY environment variables."
        )
        raise ValueError(msg)
    ...
```

Pros:

* `None` is the canonical Python sentinel for "not provided."
* Pydantic's type system distinguishes "configured" (`str`) from "not configured" (`None`).
* Lazy validation: only fails when Florence is actually used.
* Clear, actionable error message pointing to the specific env variables.
* Preserves Florence's opt-in nature.
* No breaking change for users who never use Florence.
* No impact on existing tests (they monkeypatch both functions).

Cons:

* Requires type narrowing (`assert secrets.endpoint is not None`) or the validation guard before passing to httpx.
* Slight inconsistency: Foundry/Search use bare `str`, CV uses `str | None`. This is justified because CV is optional while the others are required.

Verdict: **Recommended.** Cleanest semantics with proper lazy validation.

## Test Impact Analysis

### Current test approach

Tests in [tests/test_embeddings/test_image.py](tests/test_embeddings/test_image.py) monkeypatch both `get_cv_client` and `load_cv_secrets` at the module level:

```python
monkeypatch.setattr("ai_search.embeddings.image.get_cv_client", lambda: mock_client)
monkeypatch.setattr("ai_search.embeddings.image.load_cv_secrets", lambda: secrets)
```

The `secrets` mock is a `MagicMock()` with `api_version = "2024-02-01"`. This mock never constructs `AzureComputerVisionSecrets` from env vars.

### Impact assessment

| Change                                        | Test Impact                                                |
|-----------------------------------------------|------------------------------------------------------------|
| Change `endpoint`/`api_key` to `str \| None`  | No impact. Tests monkeypatch the function, never construct the model. |
| Add validation in `get_cv_client()`           | No impact. Tests monkeypatch `get_cv_client` entirely.      |
| New test for missing config error             | New test needed. Should verify `get_cv_client()` raises `ValueError` when secrets have `None` endpoint/key. |

The `test_config.py` file does not test `AzureComputerVisionSecrets` or `load_cv_secrets()` at all, so no changes are needed there.

### Existing tests pass with no modifications

All current tests in `test_image.py` and `conftest.py` remain unaffected because they mock at the function boundary rather than relying on real secrets construction.

## Recommended Approach

**Option D: `Optional[str] = None` + validation guard in `get_cv_client()`**

This approach:

1. Uses proper Python/Pydantic semantics (`None` vs empty string).
2. Preserves Florence as opt-in (no env vars needed if you do not use image embeddings).
3. Provides a clear, actionable error message when Florence is used without configuration.
4. Requires zero changes to existing tests.
5. Keeps validation lazy and colocated with the client factory.

## Code Snippets for the Fix

### Change 1: Update `AzureComputerVisionSecrets` in `config.py`

```python
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
```

### Change 2: Add validation guard in `get_cv_client()` in `clients.py`

```python
@lru_cache(maxsize=1)
def get_cv_client() -> httpx.AsyncClient:
    """Return a cached async HTTP client for Azure Computer Vision."""
    secrets = load_cv_secrets()
    if not secrets.endpoint or not secrets.api_key:
        msg = (
            "Azure Computer Vision is not configured. "
            "Set AZURE_CV_ENDPOINT and AZURE_CV_API_KEY environment variables."
        )
        raise ValueError(msg)
    return httpx.AsyncClient(
        base_url=secrets.endpoint,
        headers={"Ocp-Apim-Subscription-Key": secrets.api_key},
        timeout=30.0,
    )
```

### Change 3 (optional): Add test for misconfiguration error

```python
class TestGetCvClientValidation:
    """Test that get_cv_client raises on missing configuration."""

    def test_raises_without_cv_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should raise ValueError when CV secrets are not configured."""
        from ai_search.clients import get_cv_client

        secrets = MagicMock()
        secrets.endpoint = None
        secrets.api_key = None

        monkeypatch.setattr("ai_search.clients.load_cv_secrets", lambda: secrets)
        get_cv_client.cache_clear()

        with pytest.raises(ValueError, match="Azure Computer Vision is not configured"):
            get_cv_client()

        get_cv_client.cache_clear()
```
