"""Batch ingest sample images into Azure AI Search.

Usage:
    uv run python scripts/ingest_samples.py
    uv run python scripts/ingest_samples.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Force system SSL certs to avoid venv certifi corruption issues.
if not os.environ.get("SSL_CERT_FILE"):
    _sys_cert = "/private/etc/ssl/cert.pem"
    if os.path.exists(_sys_cert):
        os.environ["SSL_CERT_FILE"] = _sys_cert

import httpx  # noqa: E402
import structlog  # noqa: E402

from ai_search.embeddings.pipeline import generate_all_vectors
from ai_search.extraction.extractor import extract_image
from ai_search.indexing.indexer import build_search_document, upload_documents
from ai_search.ingestion.loader import ImageInput
from ai_search.models import SearchDocument

logger = structlog.get_logger(__name__)

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "sample_images.json"

# Delay between images to stay within embed-v-4-0 rate limits.
INTER_IMAGE_DELAY_S = 10
MAX_RETRIES = 5
RETRY_BACKOFF_S = 30


def download_images(inputs: list[ImageInput]) -> dict[str, bytes]:
    """Pre-download all images synchronously to avoid async SSL issues."""
    downloaded: dict[str, bytes] = {}
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for inp in inputs:
            if inp.image_url:
                logger.info("Downloading image", image_id=inp.image_id)
                resp = client.get(inp.image_url)
                resp.raise_for_status()
                downloaded[inp.image_id] = resp.content
                logger.info(
                    "Downloaded",
                    image_id=inp.image_id,
                    size_bytes=len(resp.content),
                )
    return downloaded


async def process_image(
    image_input: ImageInput,
    image_bytes: bytes | None = None,
) -> SearchDocument:
    """Run the full ingestion pipeline for a single image with retry."""
    print(f"\n{'='*60}")
    print(f"  Processing: {image_input.image_id}")
    print(f"  Prompt: {image_input.generation_prompt[:80]}...")
    print(f"{'='*60}")

    # Step 1: Extraction
    print("  [1/3] Extracting image metadata via GPT-4o...")
    extraction = extract_image(image_input)
    print(f"    Semantic desc:    {extraction.semantic_description[:80]}...")
    print(f"    Structural desc:  {extraction.structural_description[:80]}...")
    print(f"    Style desc:       {extraction.style_description[:80]}...")
    print(f"    Characters:       {len(extraction.characters)}")
    print(f"    Narrative tone:   {extraction.narrative.tone}")

    # Step 2: Embeddings (with retry)
    print("  [2/3] Generating embedding vectors...")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            vectors = await generate_all_vectors(
                extraction,
                image_bytes=image_bytes,
            )
            break
        except Exception as exc:
            if "429" in str(exc) and attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_S * attempt
                logger.warning(
                    "Rate limited, retrying",
                    image_id=image_input.image_id,
                    attempt=attempt,
                    wait_seconds=wait,
                )
                await asyncio.sleep(wait)
            else:
                raise

    # Validate embedding dimensions
    expected = {"semantic": 3072, "structural": 1024, "style": 512, "image": 1024}
    actual = {
        "semantic": len(vectors.semantic_vector),
        "structural": len(vectors.structural_vector),
        "style": len(vectors.style_vector),
        "image": len(vectors.image_vector),
    }
    all_valid = True
    for name, exp_dim in expected.items():
        act_dim = actual[name]
        status = "OK" if act_dim == exp_dim else "MISMATCH"
        if status == "MISMATCH":
            all_valid = False
        print(f"    {name:12s} vector: {act_dim:5d} dims  (expected {exp_dim}) [{status}]")

    if not all_valid:
        raise ValueError(f"Dimension mismatch for {image_input.image_id}: {actual}")

    # Step 3: Build document
    print("  [3/3] Building search document...")
    doc = build_search_document(image_input, extraction, vectors)
    print(f"    Document fields: {sorted(doc.model_dump().keys())}")
    return doc


async def run(dry_run: bool = False, force: bool = False) -> None:
    """Load sample images and ingest them.

    Args:
        dry_run: List images without calling APIs.
        force: Re-index all images, even those already in the index.
    """
    data = json.loads(DATA_FILE.read_text())
    logger.info("Loaded sample images", count=len(data))

    inputs = [
        ImageInput(
            image_id=item["image_id"],
            generation_prompt=item["generation_prompt"],
            image_url=item.get("image_url"),
        )
        for item in data
    ]

    if dry_run:
        for inp in inputs:
            print(f"  [{inp.image_id}] {inp.generation_prompt[:60]}...")
        print(f"\nDry run complete — {len(inputs)} images would be ingested.")
        return

    # Pre-download all images synchronously (avoids async SSL issues)
    image_data = download_images(inputs)

    # Check which docs are already indexed to skip them
    already_indexed: set[str] = set()
    if not force:
        from ai_search.clients import get_search_client
        search_client = get_search_client()
        for inp in inputs:
            try:
                search_client.get_document(key=inp.image_id)
                already_indexed.add(inp.image_id)
            except Exception:
                pass
        if already_indexed:
            logger.info("Skipping already indexed", ids=sorted(already_indexed))

    success_count = 0
    for i, inp in enumerate(inputs):
        if inp.image_id in already_indexed:
            logger.info("Skipping", image_id=inp.image_id, reason="already indexed")
            success_count += 1
            continue

        logger.info("Processing", image_id=inp.image_id, progress=f"{i + 1}/{len(inputs)}")
        doc = await process_image(inp, image_bytes=image_data.get(inp.image_id))
        logger.info("Processed, uploading", image_id=inp.image_id)
        uploaded = upload_documents([doc])
        if uploaded:
            success_count += 1
            logger.info("Indexed", image_id=inp.image_id)
        else:
            logger.error("Failed to index", image_id=inp.image_id)

        # Throttle to avoid rate limits on embed-v-4-0
        if i < len(inputs) - 1:
            logger.info("Throttling", delay_seconds=INTER_IMAGE_DELAY_S)
            await asyncio.sleep(INTER_IMAGE_DELAY_S)

    print(f"\nDone — {success_count}/{len(inputs)} documents indexed successfully.")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Ingest sample images")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List images without calling APIs",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-index all images, even those already in the index",
    )
    args = parser.parse_args()

    asyncio.run(run(dry_run=args.dry_run, force=args.force))


if __name__ == "__main__":
    sys.exit(main() or 0)
