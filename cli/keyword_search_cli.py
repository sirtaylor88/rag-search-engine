"""CLI for keyword-based movie search using BM25."""

from cli.inverted_index import InvertedIndex
from cli.commands import SearchCommand, BuildCommand
from cli.commands import register_commands


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate search command."""

    parser = register_commands()
    args = parser.parse_args()
    inverted_index = InvertedIndex()

    match args.command:
        case "search":
            SearchCommand.run(args, inverted_index)

        case "build":
            BuildCommand.run(args, inverted_index)

        case _:
            parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
