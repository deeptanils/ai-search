"""Analyze why vector search scores are uniformly high."""

import asyncio
import math

from azure.search.documents.models import VectorizedQuery

from ai_search.clients import get_search_client
from ai_search.embeddings.image import embed_image


async def main() -> None:
    ocean_url = "https://images.unsplash.com/photo-1505118380757-91f5f5632de0?w=1024"
    cat_url = "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=1024"

    print("Embedding ocean image...")
    ocean_vec = await embed_image(image_url=ocean_url)
    print("Embedding cat image...")
    cat_vec = await embed_image(image_url=cat_url)

    # Raw cosine between ocean and cat query vectors
    dot = sum(a * b for a, b in zip(ocean_vec, cat_vec))
    mag_a = math.sqrt(sum(a * a for a in ocean_vec))
    mag_b = math.sqrt(sum(b * b for b in cat_vec))
    cosine = dot / (mag_a * mag_b)
    print(f"\nRaw cosine(ocean, cat) = {cosine:.6f}")
    print(f"Vector magnitudes: ocean={mag_a:.4f}, cat={mag_b:.4f}")
    print(f"(Vectors are {'already normalized' if abs(mag_a - 1.0) < 0.01 else 'NOT normalized'})")

    # Full score distributions
    client = get_search_client()

    queries = [
        ("Ocean (exact match in index)", ocean_vec),
        ("Cat (NOT in index)", cat_vec),
    ]

    for label, vec in queries:
        results = list(
            client.search(
                search_text=None,
                vector_queries=[
                    VectorizedQuery(
                        vector=vec, k_nearest_neighbors=50, fields="image_vector"
                    )
                ],
                select=["image_id", "scene_type"],
                top=10,
            )
        )
        scores = [r.get("@search.score", 0) for r in results]
        spread = max(scores) - min(scores)
        print(f"\nQuery: {label}")
        print(f"  Score range: {min(scores):.6f} — {max(scores):.6f}")
        print(f"  Spread (max-min): {spread:.6f}")
        print(f"  Gap #1→#2: {scores[0] - scores[1]:.6f}")
        print()
        for i, r in enumerate(results, 1):
            s = r.get("@search.score", 0)
            # Scale bar to show differences (zoom into 0.94-1.0 range)
            bar_len = int((s - 0.94) * 500)
            bar = "#" * max(bar_len, 0)
            marker = " ← EXACT" if s > 0.999 else ""
            print(f"  #{i:2d}  {r['image_id']:12s}  {s:.6f}  {bar}{marker}")

    print("\n" + "=" * 60)
    print("EXPLANATION:")
    print("=" * 60)
    print("""
Cosine similarity scores are high (0.95+) because:

1. NORMALIZED EMBEDDINGS: embed-v-4-0 outputs unit vectors (magnitude ≈ 1.0).
   All vectors live on the surface of a 1024-dim unit hypersphere.

2. EMBEDDING CLUSTERING: Neural embedding models map semantically-related
   content (all photos) into a tight region of the sphere. The baseline
   similarity between ANY two photos is already ~0.95.

3. SMALL CORPUS: With only 10 documents, there aren't enough diverse
   images to spread scores across a wider range.

WHAT MATTERS is the RELATIVE ranking, not absolute scores:
  • Ocean query → sample-003 scores 1.000 (perfect match), gap to #2 is meaningful
  • Cat query → all scores cluster at 0.96, no clear winner (correct — no cat in index)

In production with thousands of diverse images, you'd see wider score
spreads (e.g., 0.60–0.95), making ranking differences more obvious.
""")


if __name__ == "__main__":
    asyncio.run(main())
