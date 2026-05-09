"""CLI for keyword-based movie search using BM25."""

from argparse import ArgumentParser

from cli.commands import (
    BuildCommand,
    ComputeBM25IDFCommand,
    ComputeBM25TFCommand,
    ComputeIDFCommand,
    ComputeTFIDFCommand,
    ComputeTFCommand,
    SearchCommand,
    BM25SearchCommand,
)
from cli.schemas import (
    BM25Payload,
    BM25Request,
    SearchPayload,
    SearchRequest,
    TermPayload,
    TermRequest,
    TermWithDocIDPayload,
    TermWithDocIDRequest,
)


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate search command."""
    parser = ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # * Build
    build_cmd = BuildCommand(
        subparsers.add_parser("build", help="Build inverted index"),
    )

    #  * Compute
    tf_cmd = ComputeTFCommand(
        subparsers.add_parser(
            "tf",
            help="Get term frequency of a document",
        ),
    )
    idf_cmd = ComputeIDFCommand(
        subparsers.add_parser(
            "idf",
            help="Get inverse document frequency for a given term",
        ),
    )
    tf_idf_cmd = ComputeTFIDFCommand(
        subparsers.add_parser(
            "tfidf",
            help="Get TF-IDF score for a given term in a document",
        ),
    )
    bm25_idf_cmd = ComputeBM25IDFCommand(
        subparsers.add_parser(
            "bm25idf",
            help="Get BM25 IDF score for a given term",
        ),
    )
    bm25_tf_cmd = ComputeBM25TFCommand(
        subparsers.add_parser(
            "bm25tf",
            help="Get BM25 TF score for a given document ID and term",
        ),
    )

    # * Search
    search_cmd = SearchCommand(
        subparsers.add_parser("search", help="Search movies using BM25"),
    )
    bm25_search_cmd = BM25SearchCommand(
        subparsers.add_parser(
            "bm25search", help="Search movies using full BM25 scoring"
        ),
    )

    args = parser.parse_args()

    match args.command:
        case "build":
            build_cmd.run(TermRequest(payload=TermPayload(term=args.term)))
        case "idf":
            idf_cmd.run(TermRequest(payload=TermPayload(term=args.term)))
        case "tf":
            tf_cmd.run(
                TermWithDocIDRequest(
                    payload=TermWithDocIDPayload(doc_id=args.doc_id, term=args.term)
                )
            )
        case "tfidf":
            tf_idf_cmd.run(
                TermWithDocIDRequest(
                    payload=TermWithDocIDPayload(doc_id=args.doc_id, term=args.term)
                )
            )
        case "bm25idf":
            bm25_idf_cmd.run(TermRequest(payload=TermPayload(term=args.term)))
        case "bm25tf":
            bm25_tf_cmd.run(
                BM25Request(
                    payload=BM25Payload(
                        doc_id=args.doc_id, term=args.term, k1=args.k1, b=args.b
                    )
                )
            )
        case "search":
            search_cmd.run(
                SearchRequest(payload=SearchPayload(query=args.query, limit=args.limit))
            )
        case "bm25search":
            bm25_search_cmd.run(
                SearchRequest(payload=SearchPayload(query=args.query, limit=args.limit))
            )
        case _:
            parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
