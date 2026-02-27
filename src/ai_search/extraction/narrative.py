"""Narrative intent extraction from unified extraction output."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_search.models import ImageExtraction, NarrativeIntent


def get_narrative(extraction: ImageExtraction) -> NarrativeIntent:
    """Extract narrative intent from a completed extraction."""
    return extraction.narrative
