"""CLI for keyword-based movie search using BM25."""

from argparse import ArgumentParser

from cli.inverted_index import InvertedIndex
from cli.commands import BuildCommand, SearchCommand, FindTFCommand
from cli.commands.compute_idf_command import ComputeIDFCommand


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate search command."""
    parser = ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    inverted_index = InvertedIndex()

    search_cmd = SearchCommand(
        subparsers.add_parser("search", help="Search movies using BM25"),
        inverted_index,
    )
    build_cmd = BuildCommand(
        subparsers.add_parser("build", help="Build inverted index"),
        inverted_index,
    )
    tf_cmd = FindTFCommand(
        subparsers.add_parser("tf", help="Get term frequency of a document"),
        inverted_index,
    )
    idf_cmd = ComputeIDFCommand(
        subparsers.add_parser(
            "idf",
            help="Compute inverse document frequency of a term",
        ),
        inverted_index,
    )

    args = parser.parse_args()

    match args.command:
        case "search":
            search_cmd.run(args.query)
        case "build":
            build_cmd.run(args.data_path)
        case "tf":
            tf_cmd.run(args.doc_id, args.term)
        case "idf":
            idf_cmd.run(args.term)
        case _:
            parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
