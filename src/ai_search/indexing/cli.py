"""CLI entry point for index management."""

from __future__ import annotations

import argparse
import sys

from ai_search.indexing.schema import create_or_update_index


def main() -> None:
    """Index management CLI."""
    parser = argparse.ArgumentParser(description="AI Search Index Management")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("create", help="Create or update the search index")

    args = parser.parse_args()

    if args.command == "create":
        index = create_or_update_index()
        print(f"Index '{index.name}' created/updated successfully")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
