"""CLI entry point for image ingestion pipeline."""

from __future__ import annotations

import argparse
import asyncio
import sys

import structlog

from ai_search.embeddings.pipeline import generate_all_vectors
from ai_search.extraction.extractor import extract_image
from ai_search.indexing.indexer import build_search_document, upload_documents
from ai_search.ingestion.loader import ImageInput

logger = structlog.get_logger(__name__)


async def _process_image(image_input: ImageInput) -> None:
    """Run the full ingestion pipeline for a single image."""
    logger.info("Starting extraction", image_id=image_input.image_id)
    extraction = extract_image(image_input)

    logger.info("Generating embeddings", image_id=image_input.image_id)
    vectors = await generate_all_vectors(extraction)

    logger.info("Building search document", image_id=image_input.image_id)
    doc = build_search_document(image_input, extraction, vectors)

    logger.info("Uploading to index", image_id=image_input.image_id)
    count = upload_documents([doc])

    logger.info("Ingestion complete", image_id=image_input.image_id, uploaded=count)
    print(f"Successfully indexed image '{image_input.image_id}'")


def main() -> None:
    """Ingestion CLI entry point."""
    parser = argparse.ArgumentParser(description="AI Search Image Ingestion")
    parser.add_argument("--image-url", type=str, help="URL of the image to ingest")
    parser.add_argument("--image-file", type=str, help="Local path to the image file")
    parser.add_argument("--prompt", type=str, required=True, help="Generation prompt for the image")
    parser.add_argument("--image-id", type=str, required=True, help="Unique identifier for the image")

    args = parser.parse_args()

    if not args.image_url and not args.image_file:
        print("Error: Either --image-url or --image-file must be provided", file=sys.stderr)
        sys.exit(1)

    if args.image_url:
        image_input = ImageInput.from_url(args.image_id, args.prompt, args.image_url)
    else:
        image_input = ImageInput.from_file(args.image_id, args.prompt, args.image_file)

    asyncio.run(_process_image(image_input))


if __name__ == "__main__":
    main()
