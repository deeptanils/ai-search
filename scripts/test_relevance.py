"""Demonstrate relevance filtering using relative score analysis.

Since cosine similarity with embed-v-4-0 yields high absolute scores (0.95+)
for all inputs, we use relative metrics to determine true relevance:

1. Gap ratio: (top1 - top2) / top1  — how far ahead is the best match?
2. Z-score: (top1 - mean) / stdev  — is #1 a statistical outlier?
3. Spread: max - min across results — is there meaningful differentiation?
"""

import asyncio
import statistics

from azure.search.documents.models import VectorizedQuery

from ai_search.clients import get_search_client
from ai_search.embeddings.image import embed_image


def compute_relevance(scores: list[float]) -> dict:
    """Compute relevance metrics from a list of search scores."""
    if len(scores) < 2:
        return {"confident": False, "reason": "too few results"}

    top1 = scores[0]
    top2 = scores[1]
    mean = statistics.mean(scores)
    stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0

    gap = top1 - top2
    gap_ratio = gap / top1 if top1 > 0 else 0
    z_score = (top1 - mean) / stdev if stdev > 0 else 0
    spread = max(scores) - min(scores)

    # Confidence rules:
    # - Z-score > 2.0: top result is a statistical outlier (strong match)
    # - Gap ratio > 0.01: meaningful separation from runner-up
    # - Spread > 0.02: enough differentiation across results
    confident = z_score > 2.0 and gap_ratio > 0.01 and spread > 0.02

    return {
        "confident": confident,
        "top1_score": top1,
        "top2_score": top2,
        "gap": gap,
        "gap_ratio": gap_ratio,
        "z_score": z_score,
        "spread": spread,
        "mean": mean,
        "stdev": stdev,
    }


async def search_with_relevance(label: str, image_url: str) -> None:
    """Embed image, search, and apply relevance filtering."""
    print(f"\n{'=' * 60}")
    print(f"  Query: {label}")
    print(f"  URL: {image_url[:80]}...")
    print(f"{'=' * 60}")

    vec = await embed_image(image_url=image_url)
    client = get_search_client()

    results = list(
        client.search(
            search_text=None,
            vector_queries=[
                VectorizedQuery(
                    vector=vec, k_nearest_neighbors=50, fields="image_vector"
                )
            ],
            select=["image_id", "scene_type", "generation_prompt"],
            top=10,
        )
    )

    scores = [r.get("@search.score", 0) for r in results]
    metrics = compute_relevance(scores)

    print(f"\n  Raw scores:")
    for i, r in enumerate(results[:5], 1):
        s = r.get("@search.score", 0)
        print(f"    #{i}  {r['image_id']:12s}  {s:.6f}  {(r.get('generation_prompt') or '')[:60]}")

    print(f"\n  Relevance Metrics:")
    print(f"    Gap (#1-#2):    {metrics['gap']:.6f}  (ratio: {metrics['gap_ratio']:.4f})")
    print(f"    Z-score of #1:  {metrics['z_score']:.2f}")
    print(f"    Score spread:   {metrics['spread']:.6f}")
    print(f"    Mean ± stdev:   {metrics['mean']:.6f} ± {metrics['stdev']:.6f}")

    verdict = "RELEVANT — confident match" if metrics["confident"] else "NOT RELEVANT — no confident match"
    print(f"\n  >>> Verdict: {verdict}")

    if metrics["confident"]:
        top = results[0]
        print(f"  >>> Best match: {top['image_id']} ({top.get('scene_type', '?')})")
    else:
        print(f"  >>> Reason: z={metrics['z_score']:.2f} (need >2.0), "
              f"gap_ratio={metrics['gap_ratio']:.4f} (need >0.01), "
              f"spread={metrics['spread']:.6f} (need >0.02)")


async def main() -> None:
    test_cases = [
        # Exact match — ocean image IS sample-003
        (
            "Ocean (exact match in index)",
            "https://images.unsplash.com/photo-1505118380757-91f5f5632de0?w=1024",
        ),
        # Similar domain — different forest image
        (
            "Forest (similar to sample-006)",
            "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=1024",
        ),
        # Completely irrelevant — cat
        (
            "Cat (NOT in index)",
            "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=1024",
        ),
        # Completely irrelevant — text/code screenshot
        (
            "Abstract art (NOT in index)",
            "https://images.unsplash.com/photo-1541701494587-cb58502866ab?w=1024",
        ),
    ]

    for label, url in test_cases:
        await search_with_relevance(label, url)
        print()


if __name__ == "__main__":
    asyncio.run(main())
