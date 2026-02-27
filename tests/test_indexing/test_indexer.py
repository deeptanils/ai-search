"""Tests for the batch document indexer."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from azure.core.exceptions import HttpResponseError

from ai_search.indexing.indexer import build_search_document, upload_documents
from ai_search.models import ImageExtraction, ImageVectors, SearchDocument

if TYPE_CHECKING:
    from ai_search.ingestion.loader import ImageInput


class TestBuildSearchDocument:
    """Test the build_search_document function."""

    def test_builds_correct_document(
        self,
        sample_image_input: ImageInput,
        sample_extraction: ImageExtraction,
        sample_vectors: ImageVectors,
    ) -> None:
        """Should build a SearchDocument with flattened character vectors."""
        doc = build_search_document(sample_image_input, sample_extraction, sample_vectors)

        assert doc.image_id == "test-image-001"
        assert doc.scene_type == "urban_night"
        assert doc.character_count == 1
        assert len(doc.semantic_vector) == 3072


class TestUploadDocuments:
    """Test the upload_documents function."""

    @patch("ai_search.indexing.indexer.get_search_client")
    @patch("ai_search.indexing.indexer.load_config")
    def test_successful_upload(
        self,
        mock_config: MagicMock,
        mock_client_factory: MagicMock,
    ) -> None:
        """Should upload documents and return success count."""
        from ai_search.config import AppConfig

        mock_config.return_value = AppConfig()

        mock_client = MagicMock()
        mock_client_factory.return_value = mock_client

        result_item = MagicMock()
        result_item.succeeded = True
        mock_client.upload_documents.return_value = [result_item]

        doc = SearchDocument(
            image_id="test",
            generation_prompt="test",
            scene_type="test",
            time_of_day="day",
            lighting_condition="bright",
            primary_subject="test",
            artistic_style="test",
            tags=["test"],
            narrative_theme="test",
            narrative_type="test",
            emotional_polarity=0.0,
            low_light_score=0.5,
            character_count=0,
            metadata_json="{}",
            extraction_json="{}",
        )

        count = upload_documents([doc])
        assert count == 1

    @patch("ai_search.indexing.indexer.time.sleep")
    @patch("ai_search.indexing.indexer.get_search_client")
    @patch("ai_search.indexing.indexer.load_config")
    def test_retry_on_429(
        self,
        mock_config: MagicMock,
        mock_client_factory: MagicMock,
        mock_sleep: MagicMock,
    ) -> None:
        """Should retry on 429 status with exponential backoff."""
        from ai_search.config import AppConfig

        mock_config.return_value = AppConfig()

        mock_client = MagicMock()
        mock_client_factory.return_value = mock_client

        error = HttpResponseError(message="Rate limited")
        error.status_code = 429

        result_item = MagicMock()
        result_item.succeeded = True

        mock_client.upload_documents.side_effect = [
            error,
            [result_item],
        ]

        doc = SearchDocument(
            image_id="test",
            generation_prompt="test",
            scene_type="test",
            time_of_day="day",
            lighting_condition="bright",
            primary_subject="test",
            artistic_style="test",
            tags=["test"],
            narrative_theme="test",
            narrative_type="test",
            emotional_polarity=0.0,
            low_light_score=0.5,
            character_count=0,
            metadata_json="{}",
            extraction_json="{}",
        )

        count = upload_documents([doc])
        assert count == 1
        mock_sleep.assert_called_once()
