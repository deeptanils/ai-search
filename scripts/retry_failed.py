"""Retry ingestion for specific failed image IDs."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

if not os.environ.get("SSL_CERT_FILE"):
    _sys_cert = "/private/etc/ssl/cert.pem"
    if os.path.exists(_sys_cert):
        os.environ["SSL_CERT_FILE"] = _sys_cert

from ai_search.ingestion.pipeline import ingest  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "sample_images.json"
FAILED_IDS = {"ramayana-0218"}


async def retry() -> None:
    data = json.loads(DATA_FILE.read_text())
    subset = [item for item in data if item["image_id"] in FAILED_IDS]
    print(f"Retrying {len(subset)} failed images...")
    await ingest(subset, PROJECT_ROOT, force=True, concurrency=2)


if __name__ == "__main__":
    asyncio.run(retry())
