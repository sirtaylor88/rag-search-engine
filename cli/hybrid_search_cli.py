"""CLI for hybrid search utilities (score normalisation)."""

from argparse import ArgumentParser
import logging

from cli.commands import NormalizeCommand, RRFSearchCommand, WeightedSearchCommand
from cli.schemas import (
    RRFSearchPayload,
    RRFSearchRequest,
    ScoreListPayload,
    ScoreListRequest,
    WeightedSearchPayload,
    WeightedSearchRequest,
)


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate hybrid search command."""
    parser = ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_cmd = NormalizeCommand(
        subparsers.add_parser("normalize", help="Normalize scores via min-max scaling"),
    )
    weighted_search_cmd = WeightedSearchCommand(
        subparsers.add_parser(
            "weighted-search", help="Search movies using weighted search"
        ),
    )
    rrf_search_cmd = RRFSearchCommand(
        subparsers.add_parser(
            "rrf-search", help="Search movies using Reciprocal Rank Fusion"
        ),
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_cmd.run(
                ScoreListRequest(payload=ScoreListPayload(scores=args.args))
            )
        case "weighted-search":
            weighted_search_cmd.run(
                WeightedSearchRequest(
                    payload=WeightedSearchPayload(
                        query=args.query,
                        alpha=args.alpha,
                        limit=args.limit,
                    )
                )
            )
        case "rrf-search":
            if args.verbose:
                logging.basicConfig(level=logging.DEBUG, format="[DEBUG] %(message)s")
            rrf_search_cmd.run(
                RRFSearchRequest(
                    payload=RRFSearchPayload(
                        query=args.query,
                        k=args.k,
                        limit=args.limit,
                        enhance=args.enhance,
                        rerank_method=args.rerank_method,
                        evaluate=args.evaluate,
                    )
                )
            )
        case _:
            parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
