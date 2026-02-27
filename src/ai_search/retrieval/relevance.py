"""Relevance scoring for vector search results.

Cosine similarity with high-dimensional embeddings (e.g., embed-v-4-0)
produces uniformly high absolute scores (0.95+).  Absolute thresholds
cannot distinguish true matches from noise.

This module uses *relative* metrics across the result set to assign
tiered confidence levels:

- **HIGH**: #1 is a statistical outlier — very likely a true match.
- **MEDIUM**: #1 stands out moderately — probable match, worth surfacing.
- **LOW**: No result stands out — query has no confident match in the index.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class Confidence(str, Enum):
    """Tiered relevance confidence."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class RelevanceResult:
    """Relevance assessment for a search result set."""

    confidence: Confidence
    top_score: float
    gap: float
    gap_ratio: float
    z_score: float
    spread: float
    mean: float
    stdev: float

    @property
    def is_relevant(self) -> bool:
        """Return True for HIGH or MEDIUM confidence."""
        return self.confidence in (Confidence.HIGH, Confidence.MEDIUM)


# ---------------------------------------------------------------------------
# Thresholds — tuned for embed-v-4-0 (1024-dim, near-unit vectors)
# ---------------------------------------------------------------------------

# HIGH confidence: #1 is a clear outlier (exact or very close match)
_HIGH_Z_SCORE = 2.0
_HIGH_GAP_RATIO = 0.01
_HIGH_SPREAD = 0.02

# MEDIUM confidence: #1 is moderately distinct (similar / relative match)
# Note: With very small corpora (<50 docs), MEDIUM may produce false
# positives because all score distributions are compressed.  These
# thresholds are tuned for production corpora of 100+ documents.
_MEDIUM_Z_SCORE = 1.3
_MEDIUM_GAP_RATIO = 0.005
_MEDIUM_SPREAD = 0.015


def score_relevance(scores: list[float]) -> RelevanceResult:
    """Assess relevance from a ranked list of search scores.

    Args:
        scores: Descending-sorted search scores from a vector query.

    Returns:
        A ``RelevanceResult`` with confidence tier and metrics.
    """
    if len(scores) < 2:
        return RelevanceResult(
            confidence=Confidence.LOW,
            top_score=scores[0] if scores else 0.0,
            gap=0.0,
            gap_ratio=0.0,
            z_score=0.0,
            spread=0.0,
            mean=scores[0] if scores else 0.0,
            stdev=0.0,
        )

    top1 = scores[0]
    top2 = scores[1]
    mean = statistics.mean(scores)
    stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0

    gap = top1 - top2
    gap_ratio = gap / top1 if top1 > 0 else 0.0
    z_score = (top1 - mean) / stdev if stdev > 0 else 0.0
    spread = max(scores) - min(scores)

    # Determine confidence tier
    if (
        z_score >= _HIGH_Z_SCORE
        and gap_ratio >= _HIGH_GAP_RATIO
        and spread >= _HIGH_SPREAD
    ):
        confidence = Confidence.HIGH
    elif (
        z_score >= _MEDIUM_Z_SCORE
        and gap_ratio >= _MEDIUM_GAP_RATIO
        and spread >= _MEDIUM_SPREAD
    ):
        confidence = Confidence.MEDIUM
    else:
        confidence = Confidence.LOW

    result = RelevanceResult(
        confidence=confidence,
        top_score=top1,
        gap=gap,
        gap_ratio=gap_ratio,
        z_score=z_score,
        spread=spread,
        mean=mean,
        stdev=stdev,
    )

    logger.info(
        "Relevance scored",
        confidence=confidence.value,
        z_score=round(z_score, 3),
        gap_ratio=round(gap_ratio, 4),
        spread=round(spread, 6),
    )
    return result


def filter_by_relevance(
    documents: list[dict[str, Any]],
    score_key: str = "search_score",
    min_confidence: Confidence = Confidence.MEDIUM,
) -> tuple[list[dict[str, Any]], RelevanceResult]:
    """Filter search results by relevance confidence.

    Args:
        documents: Search result dicts with a score field.
        score_key: Key for the search score in each document.
        min_confidence: Minimum confidence to keep results.

    Returns:
        Tuple of (filtered documents, relevance assessment).
        If confidence is below ``min_confidence``, returns empty list.
    """
    if not documents:
        empty = RelevanceResult(
            confidence=Confidence.LOW,
            top_score=0.0,
            gap=0.0,
            gap_ratio=0.0,
            z_score=0.0,
            spread=0.0,
            mean=0.0,
            stdev=0.0,
        )
        return [], empty

    scores = [doc.get(score_key, 0.0) for doc in documents]
    relevance = score_relevance(scores)

    confidence_rank = {Confidence.LOW: 0, Confidence.MEDIUM: 1, Confidence.HIGH: 2}
    if confidence_rank[relevance.confidence] < confidence_rank[min_confidence]:
        logger.info(
            "Results filtered out — below confidence threshold",
            confidence=relevance.confidence.value,
            min_required=min_confidence.value,
        )
        return [], relevance

    return documents, relevance
