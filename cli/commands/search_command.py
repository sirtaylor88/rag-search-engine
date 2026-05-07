"""Search command: queries the inverted index and prints the top results."""

from argparse import Namespace
import sys
from typing import override


from cli.utils import get_stemmed_tokens
from cli.inverted_index import InvertedIndex
from cli.commands.base import Command
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


class SearchCommand(Command):  # pylint: disable=too-few-public-methods
    """Command that loads the cached index and prints the top matching movies."""

    @override
    @staticmethod
    def run(args: Namespace, inverted_index: InvertedIndex) -> None:  # pylint: disable=arguments-differ
        """Load the index from cache and display the best matching results for the query.

        Args:
            args (Namespace): Parsed CLI arguments; uses args.query.
            inverted_index (InvertedIndex): Index instance to load cache into.
        """
        try:
            inverted_index.load()
        except OSError:
            print("Cannot load movies data.")
            sys.exit(1)

        search_query = args.query

        print("Searching for:", search_query)

        display_best_results(search_query, inverted_index)
