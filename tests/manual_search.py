#!/usr/bin/env python3
"""Dump a server search and its per-channel candidate rankings as JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests


def add_result_ranks(data: dict) -> None:
    """Put the channel ranks next to each returned result as well as in candidates."""
    candidates = {
        (collection["collection_name"], candidate["chunk_id"]): candidate
        for collection in data["collections"]
        for candidate in collection["candidates"]
    }
    results = []
    for result in data["results"]:
        candidate = candidates[(result["collection_name"], result["chunk_id"])]
        results.append({
            "semantic_rank": candidate["semantic_rank"],
            "keyword_rank": candidate["keyword_rank"],
            "path_rank": candidate["path_rank"],
            **result,
        })
    data["results"] = results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Search question or keywords")
    parser.add_argument("collection", help="Collection name, or all")
    parser.add_argument("-o", "--output", type=Path, help="Write UTF-8 JSON to a file")
    parser.add_argument("--server", default="http://localhost:8000", help="Filechatter server URL")
    parser.add_argument("--source", help="Restrict the search to an indexed path or filename")
    args = parser.parse_args()

    try:
        response = requests.post(
            f"{args.server.rstrip('/')}/search/debug",
            json={"question": args.question, "collection_name": args.collection, "source": args.source},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        add_result_ranks(data)
    except (requests.RequestException, ValueError) as exc:
        print(f"Search failed: {exc}", file=sys.stderr)
        return 1
    except (KeyError, TypeError) as exc:
        print(f"Search failed: Incomplete or invalid search-debug response: {exc}", file=sys.stderr)
        return 1

    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        print(text, end="")
    else:
        try:
            args.output.write_text(text, encoding="utf-8")
        except OSError as exc:
            print(f"Cannot write output file: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
