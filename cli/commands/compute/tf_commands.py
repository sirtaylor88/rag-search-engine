"""Term-frequency command: looks up how often a term appears in a document."""

from argparse import ArgumentParser
from typing import override

from cli.commands.base import TermCommand
from cli.schemas import BM25Payload, Request, TermWithDocIDPayload
from cli.constants import BM25_B, BM25_K1


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
    def run(self, request: Request[BM25Payload]) -> None:  # type: ignore[override]
        """Load the index from cache and print the BM25 TF score for a document.

        Args:
            request (Request[BM25Payload]): Contains the document ID,
                term, and BM25 k1 and b parameters.
        """
        self.load_cache()

        bm25tf = self.inverted_index.get_bm25_tf(
            request.payload.doc_id,
            request.payload.term,
            k1=request.payload.k1,
            b=request.payload.b,
        )
        print(
            f"BM25 TF score of '{request.payload.term}' in document "
            f"'{request.payload.doc_id}': {bm25tf:.2f}"
        )


class ComputeTFIDFCommand(ComputeTFCommand):
    """Command that loads the cached index and prints the TF-IDF for a term."""

    @override
    def run(self, request: Request[TermWithDocIDPayload]) -> None:
        """Load the index from cache and print the TF-IDF score for the given document.

        Args:
            request (Request[TermWithDocIDPayload]): Contains the document ID and term.
        """
        self.load_cache()

        idf = self.inverted_index.get_idf(request.payload.term)
        tf = self.inverted_index.get_tf(request.payload.doc_id, request.payload.term)

        tf_idf = tf * idf

        print(
            f"TF-IDF score of '{request.payload.term}' in document "
            f"'{request.payload.doc_id}': {tf_idf:.2f}"
        )
