"""Document batch uploader with retry logic for Azure AI Search."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

import structlog
from azure.core.exceptions import HttpResponseError

from ai_search.clients import get_search_client
from ai_search.config import load_config
from ai_search.models import ImageExtraction, ImageVectors, SearchDocument

if TYPE_CHECKING:
    from ai_search.ingestion.loader import ImageInput

logger = structlog.get_logger(__name__)


def build_search_document(
    image_input: ImageInput,
    extraction: ImageExtraction,
    vectors: ImageVectors,
) -> SearchDocument:
    """Assemble a SearchDocument from pipeline outputs."""
    metadata = extraction.metadata

    return SearchDocument(
        image_id=image_input.image_id,
        generation_prompt=image_input.generation_prompt,
        scene_type=metadata.scene_type,
        time_of_day=metadata.time_of_day,
        lighting_condition=metadata.lighting_condition,
        primary_subject=metadata.primary_subject,
        artistic_style=metadata.artistic_style,
        tags=metadata.tags,
        narrative_theme=metadata.narrative_theme,
        narrative_type=extraction.narrative.narrative_type,
        emotional_polarity=extraction.emotion.emotional_polarity,
        low_light_score=extraction.low_light.brightness_score,
        character_count=len(extraction.characters),
        metadata_json=json.dumps(metadata.model_dump()),
        extraction_json=json.dumps(extraction.model_dump()),
        semantic_vector=vectors.semantic_vector,
        structural_vector=vectors.structural_vector,
        style_vector=vectors.style_vector,
        image_vector=vectors.image_vector,
    )


def upload_documents(
    documents: list[SearchDocument],
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> int:
    """Upload documents in batches with exponential backoff retry.

    Returns:
        Number of successfully uploaded documents.
    """
    config = load_config()
    client = get_search_client()
    batch_size = config.batch.index_batch_size
    total_uploaded = 0

    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        docs = [_prepare_document(doc) for doc in batch]

        for attempt in range(max_retries):
            try:
                result = client.upload_documents(documents=docs)
                succeeded = sum(1 for r in result if r.succeeded)
                total_uploaded += succeeded
                logger.info(
                    "Batch uploaded",
                    batch_start=i,
                    batch_size=len(batch),
                    succeeded=succeeded,
                )
                break
            except HttpResponseError as e:
                if e.status_code in (429, 503) and attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        "Retrying batch upload",
                        status_code=e.status_code,
                        attempt=attempt + 1,
                        delay=delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error("Batch upload failed", error=str(e), batch_start=i)
                    raise

    return total_uploaded


def _prepare_document(doc: SearchDocument) -> dict[str, object]:
    """Convert SearchDocument to dict, omitting empty vector fields."""
    data = doc.model_dump()
    return {k: v for k, v in data.items() if not (isinstance(v, list) and len(v) == 0)}
