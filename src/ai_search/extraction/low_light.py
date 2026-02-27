"""Low-light metrics extraction from unified extraction output."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_search.models import ImageExtraction, LowLightMetrics


def get_low_light(extraction: ImageExtraction) -> LowLightMetrics:
    """Extract low-light metrics from a completed extraction."""
    return extraction.low_light
