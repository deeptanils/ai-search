"""Test tiered relevance filtering with exact, similar, and irrelevant images.

Demonstrates three confidence tiers:
  HIGH   — exact or near-exact match (e.g., same image in index)
  MEDIUM — relative/similar match (e.g., different forest photo)
  LOW    — no relevant match (e.g., cat photo, abstract art)
"""

import asyncio

from azure.search.documents.models import VectorizedQuery

from ai_search.clients import get_search_client
from ai_search.embeddings.image import embed_image
from ai_search.retrieval.relevance import Confidence, score_relevance


async def search_and_score(label: str, image_url: str, top: int = 10) -> None:
    """Embed image, run vector search, and score relevance."""
    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print(f"  {image_url[:80]}")
    print(f"{'=' * 65}")

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
            top=top,
        )
    )

    scores = [r.get("@search.score", 0) for r in results]
    relevance = score_relevance(scores)

    # Show top 5 results
    for i, r in enumerate(results[:5], 1):
        s = r.get("@search.score", 0)
        marker = ""
        if i == 1 and relevance.confidence == Confidence.HIGH:
            marker = "  ← STRONG MATCH"
        elif i == 1 and relevance.confidence == Confidence.MEDIUM:
            marker = "  ← PROBABLE MATCH"
        print(f"  #{i}  {r['image_id']:12s}  {s:.6f}  {(r.get('generation_prompt') or '')[:55]}{marker}")

    # Metrics
    print(f"\n  z-score:   {relevance.z_score:.2f}   (HIGH≥2.0, MED≥1.2)")
    print(f"  gap_ratio: {relevance.gap_ratio:.4f} (HIGH≥0.01, MED≥0.003)")
    print(f"  spread:    {relevance.spread:.6f} (HIGH≥0.02, MED≥0.01)")

    conf = relevance.confidence.value.upper()
    emoji = {"HIGH": "✅", "MEDIUM": "🟡", "LOW": "❌"}[conf]
    print(f"\n  {emoji} Confidence: {conf}")
    if relevance.is_relevant:
        print(f"  → Best match: {results[0]['image_id']} ({results[0].get('scene_type', '?')})")
    else:
        print(f"  → No confident match — results would be filtered out")


async def main() -> None:
    test_cases = [
        # 1. EXACT match — this image IS in the index as sample-003
        (
            "TEST 1: Exact match (ocean waves = sample-003)",
            "https://images.unsplash.com/photo-1505118380757-91f5f5632de0?w=1024",
        ),
        # 2. SIMILAR — a different mountain/snow image, should relate to sample-009
        (
            "TEST 2: Similar match (different mountain photo → sample-009?)",
            "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=1024",
        ),
        # 3. SIMILAR — a different night city scene, should relate to sample-001
        (
            "TEST 3: Similar match (different neon city → sample-001?)",
            "https://images.unsplash.com/photo-1545893835-abaa50cbe628?w=1024",
        ),
        # 4. IRRELEVANT — a cat, nothing like this in the index
        (
            "TEST 4: Irrelevant (cat — nothing in index)",
            "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=1024",
        ),
        # 5. IRRELEVANT — abstract digital art
        (
            "TEST 5: Irrelevant (abstract art — nothing in index)",
            "https://images.unsplash.com/photo-1541701494587-cb58502866ab?w=1024",
        ),
    ]

    for label, url in test_cases:
        await search_and_score(label, url)
    print()


if __name__ == "__main__":
    asyncio.run(main())
