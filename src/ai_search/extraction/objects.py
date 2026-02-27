"""Required objects extraction from unified extraction output."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_search.models import ImageExtraction, RequiredObjects


def get_objects(extraction: ImageExtraction) -> RequiredObjects:
    """Extract required objects from a completed extraction."""
    return extraction.objects
