"""Term-frequency command: looks up how often a term appears in a document."""

from argparse import ArgumentParser
from typing import override

from cli.commands.base import Request, TermCommand, TermWithDocIDPayload


class ComputeTFCommand(TermCommand):
    """Command that loads the cached index and prints the term frequency for a doc."""

    term_help = "Term to get term frequency for"

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Register doc_id and term positional arguments with the tf subparser.

        Args:
            parser (ArgumentParser): The tf subparser.
        """
        parser.add_argument("doc_id", type=int, help="Document ID")
        super().add_arguments(parser)

    @override
    def run(self, request: Request[TermWithDocIDPayload]) -> None:
        """Load the index from cache and print the term frequency for a document.

        Args:
            request (Request[TermWithDocIDPayload]): Contains the document ID and term.
        """
        self.load_cache()

        tf = self.inverted_index.get_tf(request.payload.doc_id, request.payload.term)
        print(
            f"The term frequency of ``{request.payload.term}`` "
            f"in document {request.payload.doc_id} is",
            tf,
        )
