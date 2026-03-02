"""Debug image search quality — inspect GPT-4o descriptions and results."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

if not os.environ.get("SSL_CERT_FILE"):
    _sys_cert = "/private/etc/ssl/cert.pem"
    if os.path.exists(_sys_cert):
        os.environ["SSL_CERT_FILE"] = _sys_cert

from ai_search.models import SearchMode
from ai_search.retrieval.query import _extract_image_descriptions, generate_image_query_vectors
from ai_search.retrieval.search import execute_vector_search


async def main() -> None:
    import time

    img_path = Path("h4.png")
    print(f"Testing image search with: {img_path.name}")
    img = img_path.read_bytes()

    # 1) Check GPT-4o descriptions (timed)
    print("\n--- GPT-4o Extracted Descriptions ---")
    t0 = time.perf_counter()
    descs = await _extract_image_descriptions(img)
    t_gpt = time.perf_counter() - t0
    print(f"⏱  GPT-4o extraction: {t_gpt:.2f}s")
    print(f"\n=== SEMANTIC ({len(descs.semantic_description)} chars) ===")
    print(descs.semantic_description)
    print(f"\n=== STRUCTURAL ({len(descs.structural_description)} chars) ===")
    print(descs.structural_description)
    print(f"\n=== STYLE ({len(descs.style_description)} chars) ===")
    print(descs.style_description)

    # 2) Full image search (timed end-to-end)
    print("\n--- Generating all 4 query vectors ---")
    t1 = time.perf_counter()
    vectors = await generate_image_query_vectors(img)
    t_vecs = time.perf_counter() - t1
    print(f"⏱  Total vector generation: {t_vecs:.2f}s")
    print(f"Vector keys: {list(vectors.keys())}")
    print(f"Vector dims: { {k: len(v) for k, v in vectors.items()} }")

    results = execute_vector_search(query_vectors=vectors, top=10, search_mode=SearchMode.IMAGE)
    print(f"\n--- Search Results (top 10) — image_weight=0.65 ---")
    for i, doc in enumerate(results, 1):
        print(
            f"  {i}. [{doc['image_id']}] score={doc['search_score']:.4f}  "
            f"action={doc.get('character_action', '')}  "
            f"location={doc.get('location_name', '')}  "
            f"episode={doc.get('episode_name', '')}  "
            f"scene={doc.get('scene_type', '')}  "
            f"tags={doc.get('tags', [])[:5]}"
        )

    # 3) Also try with ONLY image_vector (Cohere embedding) to test that path
    print("\n--- Image-only vector search (Cohere embed only) ---")
    img_only = {"image_vector": vectors["image_vector"]}
    results_img = execute_vector_search(query_vectors=img_only, top=5, search_mode=SearchMode.IMAGE)
    for i, doc in enumerate(results_img, 1):
        print(
            f"  {i}. [{doc['image_id']}] score={doc['search_score']:.4f}  "
            f"action={doc.get('character_action', '')}  "
            f"episode={doc.get('episode_name', '')}  "
            f"scene={doc.get('scene_type', '')}"
        )

    # 4) Also try with ONLY text vectors (semantic+structural+style)
    print("\n--- Text-only vectors search (3 text embeddings from descriptions) ---")
    text_only = {k: v for k, v in vectors.items() if k != "image_vector"}
    results_txt = execute_vector_search(query_vectors=text_only, top=5)
    for i, doc in enumerate(results_txt, 1):
        print(
            f"  {i}. [{doc['image_id']}] score={doc['search_score']:.4f}  "
            f"action={doc.get('character_action', '')}  "
            f"episode={doc.get('episode_name', '')}  "
            f"scene={doc.get('scene_type', '')}"
        )


if __name__ == "__main__":
    asyncio.run(main())
