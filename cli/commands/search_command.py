"""Search command: queries the inverted index and prints the top results."""
# pylint: disable=duplicate-code

from argparse import ArgumentParser
import sys
from typing import override

from cli.utils import get_stemmed_tokens
from cli.inverted_index import InvertedIndex
from cli.commands.base import BaseCommand
from cli.constants import STOP_WORDS


def display_best_results(
    search_query: str,
    inverted_index: InvertedIndex,
    nb_of_results: int = 5,
) -> None:
    """Print the top matching movies whose titles share a stemmed token with the query
    (case-insensitive, punctuation-insensitive, stop-words excluded).

    Args:
        search_query (str): The query string to match against movie titles.
        inverted_index (InvertedIndex): The pre-built inverted index to search.
        nb_of_results (int): Maximum number of results to display. Defaults to 5.
    """
    query_tokens = set(get_stemmed_tokens(search_query)) - set(STOP_WORDS)
    doc_ids: set[int] = set()
    for query_token in query_tokens:
        if len(doc_ids) >= nb_of_results:
            break
        doc_ids.update(inverted_index.get_documents(query_token))

    for doc_id in sorted(doc_ids)[:nb_of_results]:
        title = inverted_index.docmap[doc_id]["title"]
        print(f"{doc_id}. {title}")


class SearchCommand(BaseCommand):
    """Command that loads the cached index and prints the top matching movies."""

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Register the query positional argument with the search subparser.

        Args:
            parser (ArgumentParser): The search subparser.
        """
        parser.add_argument("query", type=str, help="Search query")

    @override
    def run(self, search_query: str) -> None:  # pylint: disable=arguments-differ
        """Load the index from cache and display the best matching results.

        Args:
            search_query (str): The search query string.
        """
        try:
            self.inverted_index.load()
        except OSError:
            print("Cannot load movies data. Please run build command first.")
            sys.exit(1)

        print("Searching for:", search_query)

        display_best_results(search_query, self.inverted_index)
