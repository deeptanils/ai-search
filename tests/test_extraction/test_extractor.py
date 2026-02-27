"""Tests for the unified GPT-4o vision extractor."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from ai_search.extraction.extractor import extract_image
from ai_search.models import ImageExtraction

if TYPE_CHECKING:
    from ai_search.ingestion.loader import ImageInput


class TestExtractImage:
    """Test the extract_image function."""

    @patch("ai_search.extraction.extractor.get_openai_client")
    @patch("ai_search.extraction.extractor.load_config")
    def test_calls_gpt4o_with_correct_format(
        self,
        mock_config: MagicMock,
        mock_client_factory: MagicMock,
        sample_image_input: ImageInput,
        sample_extraction: ImageExtraction,
    ) -> None:
        """Verify the extraction call uses structured output with ImageExtraction schema."""
        mock_client = MagicMock()
        mock_client_factory.return_value = mock_client

        config = MagicMock()
        config.models.llm_model = "gpt-4o"
        config.extraction.temperature = 0.2
        config.extraction.max_tokens = 4096
        mock_config.return_value = config

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.parsed = sample_extraction
        mock_client.beta.chat.completions.parse.return_value = response

        result = extract_image(sample_image_input)

        assert result is sample_extraction
        mock_client.beta.chat.completions.parse.assert_called_once()

        call_kwargs = mock_client.beta.chat.completions.parse.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o"
        assert call_kwargs.kwargs["response_format"] is ImageExtraction

    @patch("ai_search.extraction.extractor.get_openai_client")
    @patch("ai_search.extraction.extractor.load_config")
    def test_raises_on_none_parsed(
        self,
        mock_config: MagicMock,
        mock_client_factory: MagicMock,
        sample_image_input: ImageInput,
    ) -> None:
        """Verify ValueError raised when parsed is None."""
        mock_client = MagicMock()
        mock_client_factory.return_value = mock_client

        config = MagicMock()
        config.models.llm_model = "gpt-4o"
        config.extraction.temperature = 0.2
        config.extraction.max_tokens = 4096
        mock_config.return_value = config

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.parsed = None
        mock_client.beta.chat.completions.parse.return_value = response

        with pytest.raises(ValueError, match="no parsed extraction"):
            extract_image(sample_image_input)
