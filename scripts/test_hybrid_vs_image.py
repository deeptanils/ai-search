"""Compare image-only vs hybrid search for relative matching.

Shows that combining text + multiple vectors dramatically improves
relevance detection for similar (non-exact) queries.
"""

import asyncio

from azure.search.documents.models import VectorizedQuery

from ai_search.clients import get_search_client
from ai_search.embeddings.image import embed_image, embed_text_for_image_search
from ai_search.embeddings.encoder import embed_text
from ai_search.retrieval.relevance import Confidence, score_relevance


async def _embed_query_text(text: str, dims: int) -> list[float]:
    """Embed a text query via text-embedding-3-large."""
    return await embed_text(text, dimensions=dims)


async def search_image_only(label: str, image_url: str) -> None:
    """Pure image-vector search."""
    vec = await embed_image(image_url=image_url)
    client = get_search_client()
    results = list(
        client.search(
            search_text=None,
            vector_queries=[
                VectorizedQuery(vector=vec, k_nearest_neighbors=50, fields="image_vector")
            ],
            select=["image_id", "scene_type"],
            top=10,
        )
    )
    scores = [r.get("@search.score", 0) for r in results]
    rel = score_relevance(scores)
    top = results[0]
    print(f"  Image-only:  #{1} {top['image_id']:12s} ({top.get('scene_type','?'):20s})  "
          f"z={rel.z_score:.2f}  gap={rel.gap_ratio:.4f}  → {rel.confidence.value.upper()}")


async def search_hybrid(label: str, query_text: str, image_url: str | None = None) -> None:
    """Hybrid: text + semantic + structural + style + image vectors."""
    # Build text vectors (semantic=3072, structural=1024, style=512)
    sem, struc, sty = await asyncio.gather(
        _embed_query_text(query_text, 3072),
        _embed_query_text(query_text, 1024),
        _embed_query_text(query_text, 512),
    )

    vector_queries = [
        VectorizedQuery(vector=sem, k_nearest_neighbors=50, fields="semantic_vector", weight=4.0),
        VectorizedQuery(vector=struc, k_nearest_neighbors=50, fields="structural_vector", weight=1.5),
        VectorizedQuery(vector=sty, k_nearest_neighbors=50, fields="style_vector", weight=1.5),
    ]

    # Add image vector if an image URL is provided
    if image_url:
        img_vec = await embed_image(image_url=image_url)
        vector_queries.append(
            VectorizedQuery(vector=img_vec, k_nearest_neighbors=50, fields="image_vector", weight=2.0)
        )

    client = get_search_client()
    results = list(
        client.search(
            search_text=query_text,
            vector_queries=vector_queries,
            select=["image_id", "scene_type"],
            top=10,
        )
    )
    scores = [r.get("@search.score", 0) for r in results]
    rel = score_relevance(scores)
    top = results[0]
    vecs_used = "text+4vec" if image_url else "text+3vec"
    print(f"  Hybrid({vecs_used}): #{1} {top['image_id']:12s} ({top.get('scene_type','?'):20s})  "
          f"z={rel.z_score:.2f}  gap={rel.gap_ratio:.4f}  → {rel.confidence.value.upper()}")


async def main() -> None:
    test_cases = [
        {
            "label": "TEST 1: Exact match — ocean waves (= sample-003)",
            "text": "aerial ocean waves crashing against volcanic rocks",
            "image": "https://images.unsplash.com/photo-1505118380757-91f5f5632de0?w=1024",
        },
        {
            "label": "TEST 2: Relative — snow mountain (≈ sample-009)",
            "text": "snow covered mountain peak at dawn with dramatic clouds",
            "image": "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=1024",
        },
        {
            "label": "TEST 3: Relative — neon city night (≈ sample-001)",
            "text": "neon city street at night with reflections on wet pavement",
            "image": "https://images.unsplash.com/photo-1545893835-abaa50cbe628?w=1024",
        },
        {
            "label": "TEST 4: Irrelevant — cat photo",
            "text": "cute cat sitting on a windowsill",
            "image": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=1024",
        },
    ]

    for tc in test_cases:
        print(f"\n{'=' * 75}")
        print(f"  {tc['label']}")
        print(f"{'=' * 75}")
        await search_image_only(tc["label"], tc["image"])
        await search_hybrid(tc["label"], tc["text"], tc["image"])
        await search_hybrid(tc["label"], tc["text"], image_url=None)  # text-only hybrid
        print()


if __name__ == "__main__":
    asyncio.run(main())
