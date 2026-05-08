"""BM25 TF command: computes the Okapi BM25 term frequency for a term in a document."""

from argparse import ArgumentParser
from typing import override

from cli.commands.base import BM25Request
from cli.commands.compute.compute_tf_command import ComputeTFCommand
from cli.constants import BM25_B, BM25_K1


class ComputeBM25TFCommand(ComputeTFCommand):
    """Command that loads the cached index and prints the BM25 TF score for a doc."""

    term_help = "Term to get BM25 TF score for"

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Register doc_id, term, and optional k1 arguments with the bm25tf subparser.

        Args:
            parser (ArgumentParser): The bm25tf subparser.
        """
        super().add_arguments(parser)
        parser.add_argument(
            "k1",
            type=float,
            nargs="?",
            default=BM25_K1,
            help="Tunable BM25 K1 parameter",
        )
        parser.add_argument(
            "b",
            type=float,
            nargs="?",
            default=BM25_B,
            help="Tunable BM25 b parameter",
        )

    @override
    def run(self, request: BM25Request) -> None:  # type: ignore[override]
        """Load the index from cache and print the BM25 TF score for a document.

        Args:
            request (BM25Request): Contains the document ID,
                term, and BM25 k1 and b parameters.
        """
        self.load_cache()

        bm25tf = self.inverted_index.get_bm25_tf(
            request.doc_id,
            request.term,
            k1=request.k1,
            b=request.b,
        )
        print(
            f"BM25 TF score of '{request.term}' in document "
            f"'{request.doc_id}': {bm25tf:.2f}"
        )
