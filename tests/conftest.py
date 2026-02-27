"""Shared test fixtures for the AI Search test suite."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_search.config import AppConfig
from ai_search.ingestion.loader import ImageInput
from ai_search.models import (
    CharacterDescription,
    EmotionalTrajectory,
    ImageExtraction,
    ImageMetadata,
    ImageVectors,
    LowLightMetrics,
    NarrativeIntent,
    RequiredObjects,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def sample_config(tmp_path: Path) -> AppConfig:
    """Create a sample config with defaults for testing."""
    config_data = {
        "models": {"embedding_model": "text-embedding-3-large", "llm_model": "gpt-4o"},
        "search": {
            "semantic_weight": 0.4,
            "structural_weight": 0.15,
            "style_weight": 0.15,
            "image_weight": 0.2,
            "keyword_weight": 0.1,
        },
        "index": {
            "name": "test-index",
            "vector_dimensions": {
                "semantic": 3072,
                "structural": 1024,
                "style": 512,
                "image": 1024,
            },
            "hnsw": {"m": 4, "ef_construction": 400, "ef_search": 500},
        },
        "retrieval": {
            "top_k": 50,
            "k_nearest": 100,
        },
        "extraction": {"temperature": 0.2, "max_tokens": 4096},
        "batch": {"index_batch_size": 500, "embedding_chunk_size": 2048},
    }

    config_file = tmp_path / "config.yaml"
    import yaml

    config_file.write_text(yaml.dump(config_data))

    return AppConfig(**config_data)


@pytest.fixture()
def mock_openai_client() -> MagicMock:
    """Return a mock synchronous OpenAI client."""
    client = MagicMock()

    # Mock structured output parse response
    parsed_mock = MagicMock()
    parsed_mock.choices = [MagicMock()]
    parsed_mock.choices[0].message.parsed = None
    client.beta.chat.completions.parse.return_value = parsed_mock

    # Mock regular completions
    completion_mock = MagicMock()
    completion_mock.choices = [MagicMock()]
    completion_mock.choices[0].message.content = "test response"
    client.chat.completions.create.return_value = completion_mock

    return client


@pytest.fixture()
def mock_async_openai_client() -> AsyncMock:
    """Return a mock async OpenAI client."""
    client = AsyncMock()

    # Mock embedding response
    embedding_mock = MagicMock()
    embedding_item = MagicMock()
    embedding_item.embedding = [random.random() for _ in range(256)]
    embedding_mock.data = [embedding_item]
    client.embeddings.create.return_value = embedding_mock

    return client


@pytest.fixture()
def sample_image_input() -> ImageInput:
    """Return a sample ImageInput for testing."""
    return ImageInput(
        image_id="test-image-001",
        generation_prompt="A cinematic night scene with a woman in a red dress",
        image_url="https://example.com/test-image.jpg",
    )


@pytest.fixture()
def sample_extraction() -> ImageExtraction:
    """Return a sample ImageExtraction for testing."""
    return ImageExtraction(
        semantic_description="A dramatic cinematic scene featuring a woman in a flowing red dress.",
        structural_description="The subject is centered in the frame with leading lines from shadows.",
        style_description="Film noir aesthetic with high contrast and desaturated tones.",
        characters=[
            CharacterDescription(
                character_id="woman_red_dress",
                semantic="A woman in a red dress standing in a dark alley.",
                emotion="She appears contemplative with a hint of sadness.",
                pose="Standing upright, facing slightly left, hands at sides.",
            ),
        ],
        metadata=ImageMetadata(
            scene_type="urban_night",
            time_of_day="night",
            lighting_condition="low_light",
            primary_subject="woman",
            secondary_subjects=["alley", "shadows"],
            artistic_style="film_noir",
            color_palette=["red", "black", "grey"],
            tags=["cinematic", "night", "dramatic", "film_noir"],
            narrative_theme="mystery",
        ),
        narrative=NarrativeIntent(
            story_summary="A mysterious woman walks alone through a dark alley.",
            narrative_type="cinematic",
            tone="mysterious",
        ),
        emotion=EmotionalTrajectory(
            starting_emotion="calm",
            mid_emotion="tension",
            end_emotion="resolution",
            emotional_polarity=-0.3,
        ),
        objects=RequiredObjects(
            key_objects=["woman", "red_dress"],
            contextual_objects=["alley", "streetlight"],
            symbolic_elements=["shadows", "isolation"],
        ),
        low_light=LowLightMetrics(
            brightness_score=0.2,
            contrast_score=0.8,
            noise_estimate=0.3,
            shadow_dominance=0.7,
            visibility_confidence=0.6,
        ),
    )


@pytest.fixture()
def sample_vectors() -> ImageVectors:
    """Return sample ImageVectors with random float arrays."""
    return ImageVectors(
        semantic_vector=[random.random() for _ in range(3072)],
        structural_vector=[random.random() for _ in range(1024)],
        style_vector=[random.random() for _ in range(512)],
        image_vector=[random.random() for _ in range(1024)],
    )
