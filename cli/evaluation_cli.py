"""Search evaluation CLI: measures Precision@k over a golden dataset."""

import argparse
import json
from typing import Any

from cli.constants import DEFAULT_PRECISION_AT_K
from cli.core.hybrid_search import HybridSearch
from cli.utils import load_movies


def main() -> None:
    """Run Precision@k evaluation for every test case in the golden dataset."""
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_PRECISION_AT_K,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    documents = load_movies()
    hs = HybridSearch(documents)

    with open("data/golden_dataset.json", encoding="utf-8") as fh:
        data = json.load(fh)

    test_cases: list[dict[str, Any]] = data["test_cases"]

    for test_case in test_cases:
        top_results = hs.rrf_search(test_case["query"], k=60, limit=limit)
        retrieved: set[str] = {r["title"] for r in top_results}
        test_case.update(
            {
                "retrieved": retrieved,
                "precision": len(retrieved.intersection(test_case["relevant_docs"]))
                / len(retrieved),
            }
        )

    print(f"k={limit} (Top results)")

    for test_case in sorted(
        test_cases,
        key=lambda c: c["precision"],
        reverse=True,
    ):
        print(f"- Query: {test_case['query']}")
        print(f"  - Precision@{limit}: {test_case['precision']:.4f}")
        print(f"  - Retrieved: {', '.join(test_case['retrieved'])}")
        print(f"  - Relevant: {', '.join(test_case['relevant_docs'])}")


if __name__ == "__main__":  # pragma: no cover
    main()
