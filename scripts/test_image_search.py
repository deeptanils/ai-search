"""Search the index using an image — embed it and find visually similar documents."""

from __future__ import annotations

import asyncio
import json
import sys

from azure.search.documents.models import VectorizedQuery

from ai_search.clients import get_search_client
from ai_search.embeddings.image import embed_image


async def image_search(image_url: str, top: int = 5) -> list[dict]:
    """Embed an image and search the index by image_vector similarity."""
    print(f"\n{'=' * 60}")
    print(f"  Image Search")
    print(f"  URL: {image_url[:90]}...")
    print(f"{'=' * 60}")

    # 1. Generate image embedding
    print("\n  [1/2] Embedding query image via embed-v-4-0...")
    vector = await embed_image(image_url=image_url)
    print(f"         Got {len(vector)}-dim vector")

    # 2. Vector search
    print("  [2/2] Searching index by image_vector cosine similarity...\n")
    client = get_search_client()

    results = list(
        client.search(
            search_text=None,
            vector_queries=[
                VectorizedQuery(
                    vector=vector,
                    k_nearest_neighbors=50,
                    fields="image_vector",
                ),
            ],
            select=[
                "image_id",
                "generation_prompt",
                "scene_type",
                "tags",
                "emotional_polarity",
            ],
            top=top,
        )
    )

    print(f"  Results: {len(results)}\n")
    for i, r in enumerate(results, 1):
        score = r.get("@search.score", 0)
        iid = r["image_id"]
        scene = r.get("scene_type", "?")
        prompt = (r.get("generation_prompt") or "")[:100]
        tags = r.get("tags", []) or []
        print(f"  #{i}  {iid}  (score: {score:.6f})")
        print(f"      scene: {scene}")
        print(f"      prompt: {prompt}")
        print(f"      tags: {tags[:6]}")
        print()

    return results


async def run_all() -> None:
    # Default: use one of the sample images (ocean waves — sample-003)
    # Or pass a URL as argument: python scripts/test_image_search.py <url>
    default_urls = {
        "ocean": "https://images.unsplash.com/photo-1505118380757-91f5f5632de0?w=1024",
        "forest": "https://images.unsplash.com/photo-1448375240586-882707db888b?w=1024",
        "city": "https://images.unsplash.com/photo-1542051841857-5f90071e7989?w=1024",
    }

    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = default_urls["ocean"]

    await image_search(url)

    # Also test with a completely different image not in the index
    print("\n" + "=" * 60)
    print("  Now searching with an image NOT in the index (a cat)")
    print("=" * 60)
    cat_url = "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=1024"
    await image_search(cat_url)


def main() -> None:
    asyncio.run(run_all())


if __name__ == "__main__":
    main()
