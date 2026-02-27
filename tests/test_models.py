"""Tests for Pydantic data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_search.models import (
    CharacterDescription,
    EmotionalTrajectory,
    ImageExtraction,
    ImageMetadata,
    LowLightMetrics,
    NarrativeIntent,
    RequiredObjects,
    SearchDocument,
    SearchResult,
)


class TestCharacterDescription:
    """Test CharacterDescription model."""

    def test_valid_character(self) -> None:
        char = CharacterDescription(
            character_id="woman_red_dress",
            semantic="A woman in a red dress.",
            emotion="She appears contemplative.",
            pose="Standing upright.",
        )
        assert char.character_id == "woman_red_dress"

    def test_missing_required_field(self) -> None:
        with pytest.raises(ValidationError):
            CharacterDescription(character_id="test")  # type: ignore[call-arg]


class TestEmotionalTrajectory:
    """Test EmotionalTrajectory model."""

    def test_valid_polarity_range(self) -> None:
        emotion = EmotionalTrajectory(
            starting_emotion="calm",
            mid_emotion="tension",
            end_emotion="resolution",
            emotional_polarity=-0.5,
        )
        assert emotion.emotional_polarity == -0.5

    def test_polarity_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            EmotionalTrajectory(
                starting_emotion="calm",
                mid_emotion="tension",
                end_emotion="resolution",
                emotional_polarity=1.5,
            )


class TestLowLightMetrics:
    """Test LowLightMetrics model."""

    def test_valid_scores(self) -> None:
        metrics = LowLightMetrics(
            brightness_score=0.5,
            contrast_score=0.8,
            noise_estimate=0.2,
            shadow_dominance=0.6,
            visibility_confidence=0.9,
        )
        assert metrics.brightness_score == 0.5

    def test_score_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            LowLightMetrics(
                brightness_score=1.5,
                contrast_score=0.8,
                noise_estimate=0.2,
                shadow_dominance=0.6,
                visibility_confidence=0.9,
            )


class TestImageExtraction:
    """Test ImageExtraction model."""

    def test_valid_extraction(self, sample_extraction: ImageExtraction) -> None:
        assert sample_extraction.semantic_description
        assert len(sample_extraction.characters) == 1
        assert sample_extraction.metadata.scene_type == "urban_night"

    def test_empty_characters(self) -> None:
        extraction = ImageExtraction(
            semantic_description="Test",
            structural_description="Test",
            style_description="Test",
            characters=[],
            metadata=ImageMetadata(
                scene_type="test",
                time_of_day="day",
                lighting_condition="bright",
                primary_subject="test",
                artistic_style="test",
                narrative_theme="test",
            ),
            narrative=NarrativeIntent(
                story_summary="Test",
                narrative_type="test",
                tone="test",
            ),
            emotion=EmotionalTrajectory(
                starting_emotion="test",
                mid_emotion="test",
                end_emotion="test",
                emotional_polarity=0.0,
            ),
            objects=RequiredObjects(),
            low_light=LowLightMetrics(
                brightness_score=0.5,
                contrast_score=0.5,
                noise_estimate=0.5,
                shadow_dominance=0.5,
                visibility_confidence=0.5,
            ),
        )
        assert extraction.characters == []


class TestSearchDocument:
    """Test SearchDocument model."""

    def test_default_vectors_empty(self) -> None:
        doc = SearchDocument(
            image_id="test",
            generation_prompt="test prompt",
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
        assert doc.semantic_vector == []


class TestSearchResult:
    """Test SearchResult model."""

    def test_valid_result(self) -> None:
        result = SearchResult(
            image_id="test",
            search_score=0.95,
            generation_prompt="test",
            tags=["cinematic"],
        )
        assert result.search_score == 0.95
