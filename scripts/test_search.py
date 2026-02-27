"""Quick search verification against the candidate-index."""

from ai_search.clients import get_search_client


def main() -> None:
    client = get_search_client()

    # 1. All documents
    results = list(
        client.search(
            search_text="*",
            select=[
                "image_id",
                "generation_prompt",
                "scene_type",
                "tags",
                "character_count",
                "emotional_polarity",
            ],
            top=20,
        )
    )
    print(f"=== Total documents: {len(results)} ===\n")
    for r in results:
        score = r.get("@search.score", 0)
        iid = r["image_id"]
        scene = r.get("scene_type", "?")
        chars = r.get("character_count", "?")
        pol = r.get("emotional_polarity", "?")
        prompt = (r.get("generation_prompt") or "")[:90]
        tags = r.get("tags", []) or []
        print(f"  {iid:12s}  score={score:.4f}  scene={scene}  chars={chars}  polarity={pol}")
        print(f"               prompt: {prompt}")
        print(f"               tags:   {tags[:5]}")
        print()

    # 2. Keyword: forest
    print('=== Keyword: "forest" ===')
    results = list(
        client.search(search_text="forest", select=["image_id", "generation_prompt"], top=5)
    )
    print(f"  Results: {len(results)}")
    for r in results:
        print(f"  {r['image_id']:12s}  score={r.get('@search.score',0):.4f}  {(r.get('generation_prompt') or '')[:90]}")
    print()

    # 3. Keyword: musician jazz
    print('=== Keyword: "musician jazz" ===')
    results = list(
        client.search(
            search_text="musician jazz",
            select=["image_id", "generation_prompt", "character_count"],
            top=5,
        )
    )
    print(f"  Results: {len(results)}")
    for r in results:
        print(f"  {r['image_id']:12s}  score={r.get('@search.score',0):.4f}  chars={r.get('character_count','?')}  {(r.get('generation_prompt') or '')[:90]}")
    print()

    # 4. Keyword: ocean waves
    print('=== Keyword: "ocean waves" ===')
    results = list(
        client.search(search_text="ocean waves", select=["image_id", "generation_prompt"], top=5)
    )
    print(f"  Results: {len(results)}")
    for r in results:
        print(f"  {r['image_id']:12s}  score={r.get('@search.score',0):.4f}  {(r.get('generation_prompt') or '')[:90]}")
    print()

    # 5. Filter: character_count > 0
    print("=== Filter: character_count gt 0 ===")
    results = list(
        client.search(
            search_text="*",
            filter="character_count gt 0",
            select=["image_id", "character_count", "generation_prompt"],
            top=10,
        )
    )
    print(f"  Results: {len(results)}")
    for r in results:
        print(f"  {r['image_id']:12s}  chars={r.get('character_count','?')}  {(r.get('generation_prompt') or '')[:90]}")
    print()

    # 6. Keyword: mountain snow
    print('=== Keyword: "mountain snow" ===')
    results = list(
        client.search(search_text="mountain snow", select=["image_id", "generation_prompt"], top=5)
    )
    print(f"  Results: {len(results)}")
    for r in results:
        print(f"  {r['image_id']:12s}  score={r.get('@search.score',0):.4f}  {(r.get('generation_prompt') or '')[:90]}")


if __name__ == "__main__":
    main()
