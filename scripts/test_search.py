"""Quick smoke test for text-to-image and image-to-image search."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

if not os.environ.get("SSL_CERT_FILE"):
    _sys_cert = "/private/etc/ssl/cert.pem"
    if os.path.exists(_sys_cert):
        os.environ["SSL_CERT_FILE"] = _sys_cert

from ai_search.models import SearchMode
from ai_search.retrieval.pipeline import search

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _print_results(results: list) -> None:
    for i, r in enumerate(results, 1):
        url_preview = r.image_url[:90] if r.image_url else "(none)"
        print(f"  {i}. [{r.image_id}] score={r.search_score:.4f}  scene={r.scene_type}")
        print(f"     url={url_preview}")
        print(f"     tags={r.tags[:4]}")


async def main() -> None:
    # ── Text searches ────────────────────────────────────────────────
    queries = [
        "Hanuman flying over the ocean",
        "golden palace coronation ceremony",
        "demon king battle scene",
    ]
    for q in queries:
        print(f'\n=== TEXT SEARCH: "{q}" ===')
        results = await search(mode=SearchMode.TEXT, query_text=q, top=5)
        _print_results(results)

    # ── Image search ─────────────────────────────────────────────────
    test_image = PROJECT_ROOT / "data" / "images" / "ramayana-001.png"
    if test_image.exists():
        print(f"\n=== IMAGE SEARCH: {test_image.name} ===")
        results = await search(
            mode=SearchMode.IMAGE,
            query_image_bytes=test_image.read_bytes(),
            top=5,
        )
        _print_results(results)
    else:
        print(f"\nSkipping image search — {test_image} not found")


if __name__ == "__main__":
    asyncio.run(main())
