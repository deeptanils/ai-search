"""Emotional trajectory extraction from unified extraction output."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_search.models import EmotionalTrajectory, ImageExtraction


def get_emotion(extraction: ImageExtraction) -> EmotionalTrajectory:
    """Extract emotional trajectory from a completed extraction."""
    return extraction.emotion
