"""CLI for keyword-based movie search using BM25."""

import argparse
import json
import string


def display_five_best_results(
    search_query: str,
    data_path: str = "data/movies.json",
) -> None:
    """Print the first five movies whose title contains the search query
    (case-insensitive).

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
        if search_query.lower() in remove_all_punctuations(title).lower():
            count += 1
            print(f"{count}. {title}")


def remove_all_punctuations(text: str) -> str:
    """Remove all punctuations.

    Args:
        text (str): The input string to process.

    Returns:
        str: The input string with all punctuation characters removed.
    """

    trans_table = str.maketrans("", "", string.punctuation)
    return text.translate(trans_table)


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
