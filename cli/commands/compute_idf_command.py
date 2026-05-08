"""IDF command: computes inverse document frequency for a term."""

from argparse import ArgumentParser
from typing import override

from cli.commands.base import BaseCommand, TermRequest


class ComputeIDFCommand(BaseCommand[TermRequest]):
    """Command that loads the cached index and prints the IDF for a term."""

    term_help = "Term to get IDF score for"

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Register term positional argument with the idf subparser.

        Args:
            parser (ArgumentParser): The idf subparser.
        """
        parser.add_argument("term", type=str, help=self.term_help)

    @override
    def run(self, request: TermRequest) -> None:
        """Load the index from cache and print the IDF for the given term.

        Args:
            request (TermRequest): Contains the term to compute IDF for.
        """
        self.load_cache()

        idf = self.inverted_index.get_idf(request.term)

        print(f"Inverse document frequency of '{request.term}': {idf:.2f}")
