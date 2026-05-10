"""CLI for hybrid search utilities (score normalisation)."""

from argparse import ArgumentParser

from cli.commands import NormalizeCommand, WeightedSearchCommand
from cli.schemas import (
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
        case _:
            parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
