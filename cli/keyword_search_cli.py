"""CLI for keyword-based movie search using BM25."""

import argparse
import json

from cli.constants import STOP_WORDS
from cli.utils import tokenize_text


def display_five_best_results(
    search_query: str,
    data_path: str = "data/movies.json",
) -> None:
    """Print the first five movies whose title shares a token with the search query
    (case-insensitive, punctuation-insensitive, stop-words excluded).

    Args:
        search_query (str): The query string to match against movie titles.
        data_path (str): Path to the JSON file containing the movie dataset.
    """
    with open(data_path, encoding="utf-8") as fh:
        data = json.load(fh)

    count = 0
    for movie in data["movies"]:
        if count == 5:
            break

        title = movie.get("title", "")
        query_tokens = set(tokenize_text(search_query)) - set(STOP_WORDS)
        title_tokens = tokenize_text(title)

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

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print("Searching for:", args.query)
            display_five_best_results(args.query)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
