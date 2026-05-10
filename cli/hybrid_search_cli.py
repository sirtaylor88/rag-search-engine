"""CLI for hybrid search utilities (score normalisation)."""

from argparse import ArgumentParser

from cli.commands import NormalizeCommand
from cli.schemas import ScoreListPayload, ScoreListRequest


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate hybrid search command."""
    parser = ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_cmd = NormalizeCommand(
        subparsers.add_parser("normalize", help="Normalize scores via min-max scaling"),
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_cmd.run(
                ScoreListRequest(payload=ScoreListPayload(scores=args.args))
            )
        case _:
            parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
