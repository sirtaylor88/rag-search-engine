"""Term-frequency command: looks up how often a term appears in a document."""
# pylint: disable=duplicate-code

from argparse import ArgumentParser
from typing import override

from cli.commands.base import BaseCommand


class FindTFCommand(BaseCommand):
    """Command that loads the cached index and prints the term frequency for a doc."""

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Register doc_id and term positional arguments with the tf subparser.

        Args:
            parser (ArgumentParser): The tf subparser.
        """
        parser.add_argument("doc_id", type=int, help="Document ID")
        parser.add_argument("term", type=str, help="The term")

    @override
    def run(self, doc_id: int, term: str) -> None:  # pylint: disable=arguments-differ
        """Load the index from cache and print the term frequency for a document.

        Args:
            doc_id (int): The document ID to look up.
            term (str): The term to look up.
        """
        self.load_cache()

        tf = self.inverted_index.get_tf(doc_id, term)
        print(f"The term frequency of ``{term}`` in document with ID {doc_id} is", tf)
