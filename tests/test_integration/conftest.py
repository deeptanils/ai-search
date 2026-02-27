"""Integration test fixtures — requires real Azure credentials."""

from __future__ import annotations

import os

import pytest

# Skip all integration tests unless AZURE_INTEGRATION env var is set
pytestmark = pytest.mark.integration


@pytest.fixture()
def requires_azure() -> None:
    """Skip test if Azure credentials are not configured."""
    if not os.environ.get("AZURE_FOUNDRY_ENDPOINT"):
        pytest.skip("Azure credentials not configured")
