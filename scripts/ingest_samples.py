"""Batch ingest sample images into Azure AI Search.

Usage:
    uv run python scripts/ingest_samples.py
    uv run python scripts/ingest_samples.py --dry-run
    uv run python scripts/ingest_samples.py --force --concurrency 8
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

from ai_search.ingestion.pipeline import (  # noqa: E402
    MAX_CONCURRENT,
    ingest,
    load_image_inputs,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "sample_images.json"


async def run(
    dry_run: bool = False,
    force: bool = False,
    concurrency: int = MAX_CONCURRENT,
) -> None:
    """Load sample images and ingest them."""
    data = json.loads(DATA_FILE.read_text())

    if dry_run:
        inputs, _ = load_image_inputs(data, PROJECT_ROOT)
        for inp in inputs:
            print(f"  [{inp.image_id}] {inp.generation_prompt[:60]}...")
        print(f"\nDry run complete — {len(inputs)} images would be ingested.")
        return

    await ingest(data, PROJECT_ROOT, force=force, concurrency=concurrency)


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
    parser.add_argument(
        "--concurrency",
        type=int,
        default=MAX_CONCURRENT,
        help=f"Number of images to process in parallel (default: {MAX_CONCURRENT})",
    )
    args = parser.parse_args()

    asyncio.run(run(dry_run=args.dry_run, force=args.force, concurrency=args.concurrency))


if __name__ == "__main__":
    sys.exit(main() or 0)
