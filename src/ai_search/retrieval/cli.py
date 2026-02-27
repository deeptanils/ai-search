"""CLI entry point for search queries."""

from __future__ import annotations

import argparse

from ai_search.retrieval.pipeline import retrieve_sync


def main() -> None:
    """Query CLI entry point."""
    parser = argparse.ArgumentParser(description="AI Search Query")
    parser.add_argument("--query", type=str, required=True, help="Search query text")
    parser.add_argument("--top", type=int, default=10, help="Number of results to return")
    parser.add_argument("--filter", type=str, default=None, help="OData filter expression")

    args = parser.parse_args()

    results = retrieve_sync(args.query, odata_filter=args.filter, top=args.top)

    if not results:
        print("No results found.")
        return

    print(f"\n{'='*60}")
    print(f"Query: {args.query}")
    print(f"Results: {len(results)}")
    print(f"{'='*60}\n")

    for i, result in enumerate(results, 1):
        print(f"  {i}. [{result.image_id}]")
        print(f"     Search Score:  {result.search_score:.4f}")
        if result.scene_type:
            print(f"     Scene Type:    {result.scene_type}")
        if result.generation_prompt:
            prompt_display = result.generation_prompt[:80]
            if len(result.generation_prompt) > 80:
                prompt_display += "..."
            print(f"     Prompt:        {prompt_display}")
        if result.tags:
            print(f"     Tags:          {', '.join(result.tags[:5])}")
        print()


if __name__ == "__main__":
    main()
