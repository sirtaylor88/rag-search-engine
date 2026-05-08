"""CLI for keyword-based movie search using BM25."""

from argparse import ArgumentParser

from cli.inverted_index import InvertedIndex
from cli.commands import (
    BuildCommand,
    ComputeIDFCommand,
    ComputeTFIDFCommand,
    FindTFCommand,
    SearchCommand,
    ComputeBM25IDFCommand,
)
from cli.commands.base import TermRequest, TermWithDocIDRequest


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
        subparsers.add_parser(
            "tf",
            help="Get term frequency of a document",
        ),
        inverted_index,
    )
    idf_cmd = ComputeIDFCommand(
        subparsers.add_parser(
            "idf",
            help="Get inverse document frequency for a given term",
        ),
        inverted_index,
    )
    tf_idf_cmd = ComputeTFIDFCommand(
        subparsers.add_parser(
            "tfidf",
            help="Get TF-IDF score for a given term in a document",
        ),
        inverted_index,
    )
    bm25_idf_cmd = ComputeBM25IDFCommand(
        subparsers.add_parser(
            "bm25idf",
            help="Get BM25 IDF score for a given term",
        ),
        inverted_index,
    )

    args = parser.parse_args()

    match args.command:
        case "search":
            search_cmd.run(TermRequest(term=args.query))
        case "build":
            build_cmd.run(TermRequest(term=args.term))
        case "tf":
            tf_cmd.run(TermWithDocIDRequest(doc_id=args.doc_id, term=args.term))
        case "idf":
            idf_cmd.run(TermRequest(term=args.term))
        case "tfidf":
            tf_idf_cmd.run(TermWithDocIDRequest(doc_id=args.doc_id, term=args.term))
        case "bm25idf":
            bm25_idf_cmd.run(TermRequest(term=args.term))
        case _:
            parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
