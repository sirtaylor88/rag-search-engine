"""CLI for keyword-based movie search using BM25."""

import argparse
from typing import Any

from cli.constants import STOP_WORDS
from cli.utils import get_movies, get_stemmed_tokens
from cli.inverted_index import InvertedIndex


def display_five_best_results(
    search_query: str,
    movies: list[dict[str, Any]],
) -> None:
    """Print the first five movies whose title shares a stemmed token with the query
    (case-insensitive, punctuation-insensitive, stop-words excluded).

    Args:
        search_query (str): The query string to match against movie titles.
        movies (list[dict[str, Any]]): The movies dataset.
    """

    count = 0
    for movie in movies:
        if count == 5:
            break

        title = movie.get("title", "")
        query_tokens = get_stemmed_tokens(search_query) - set(STOP_WORDS)
        title_tokens = get_stemmed_tokens(title)

        for q_token in query_tokens:
            for t_token in title_tokens:
                if q_token in t_token:
                    count += 1
                    print(f"{count}. {title}")

                    break
            else:
                continue
            break


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

    match args.command:
        case "search":
            print("Searching for:", args.query)

            movies = get_movies(data_path=args.data_path)
            display_five_best_results(args.query, movies)

        case "build":
            movies = get_movies(data_path=args.data_path)

            print("Building inverted index for", len(movies), "movies...")
            inverted_index = InvertedIndex()
            inverted_index.build(movies)
            inverted_index.save()

            print("Inverted index has been built !")

            docs = inverted_index.get_documents("merida")
            print(f"First document for token 'merida' = {docs[0]}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
