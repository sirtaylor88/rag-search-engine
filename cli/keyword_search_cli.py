"""CLI for keyword-based movie search using BM25."""

import argparse
import sys

from cli.constants import STOP_WORDS
from cli.utils import get_movies, get_stemmed_tokens
from cli.inverted_index import InvertedIndex


def display_five_best_results(
    search_query: str,
    inverted_index: InvertedIndex,
) -> None:
    """Print the first five movies whose title shares a stemmed token with the query
    (case-insensitive, punctuation-insensitive, stop-words excluded).

    Args:
        search_query (str): The query string to match against movie titles.
        inverted_index (InvertedIndex): The pre-built inverted index to search.
    """

    query_tokens = get_stemmed_tokens(search_query) - set(STOP_WORDS)
    doc_ids: set[int] = set()
    for query_token in query_tokens:
        if len(doc_ids) >= 5:
            break
        doc_ids.update(inverted_index.get_documents(query_token))

    for doc_id in sorted(doc_ids)[:5]:
        title = inverted_index.docmap[doc_id]["title"]
        print(f"{doc_id}. {title}")


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate search command."""
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # * Search command
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--data-path",
        type=str,
        default="data/movies.json",
        help="Path to the movies JSON file (default: data/movies.json)",
    )

    # * Build command
    build_parser = subparsers.add_parser("build", help="Build inverted index")
    build_parser.add_argument(
        "--data-path",
        type=str,
        default="data/movies.json",
        help="Path to the movies JSON file (default: data/movies.json)",
    )

    args = parser.parse_args()
    inverted_index = InvertedIndex()

    match args.command:
        case "search":
            try:
                inverted_index.load()
            except OSError:
                print("Cannot load movies data.")
                sys.exit(1)

            print("Searching for:", args.query)

            display_five_best_results(args.query, inverted_index)

        case "build":
            movies = get_movies(data_path=args.data_path)

            print("Building inverted index for", len(movies), "movies...")

            inverted_index.build(movies)
            inverted_index.save()

            print("Inverted index has been built !")

        case _:
            parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
