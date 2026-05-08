"""IDF command: computes inverse document frequency for a term."""

from argparse import ArgumentParser
from typing import override

from cli.commands.base import BaseCommand


class ComputeIDFCommand(BaseCommand):
    """Command that loads the cached index and prints the IDF for a term."""

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Register term positional argument with the idf subparser.

        Args:
            parser (ArgumentParser): The idf subparser.
        """
        parser.add_argument("term", type=str, help="The term")

    @override
    def run(self, term: str) -> None:  # pylint: disable=arguments-differ
        """Load the index from cache and print the IDF for the given term.

        Args:
            term (str): The term to compute IDF for.
        """
        self.load_cache()

        idf = self.inverted_index.get_idf(term)

        print(f"Inverse document frequency of '{term}': {idf:.2f}")
