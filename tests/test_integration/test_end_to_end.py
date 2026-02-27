"""End-to-end integration test placeholders.

These tests require real Azure credentials and are skipped by default.
Run with: uv run pytest tests/ -m integration
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


class TestEndToEnd:
    """End-to-end integration tests (require real Azure services)."""

    def test_placeholder_extraction_pipeline(self, requires_azure: None) -> None:
        """Placeholder: test full extraction pipeline against live GPT-4o."""
        pytest.skip("Integration test — requires Azure credentials")

    def test_placeholder_indexing_pipeline(self, requires_azure: None) -> None:
        """Placeholder: test index creation and document upload."""
        pytest.skip("Integration test — requires Azure credentials")

    def test_placeholder_retrieval_pipeline(self, requires_azure: None) -> None:
        """Placeholder: test end-to-end query pipeline."""
        pytest.skip("Integration test — requires Azure credentials")
