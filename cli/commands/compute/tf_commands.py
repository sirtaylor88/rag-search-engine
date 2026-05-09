"""TF commands: compute term frequency, TF-IDF, and BM25 TF for a term in a document."""

from abc import abstractmethod
from argparse import ArgumentParser
from typing import Generic, TypeVar, override

from cli.commands.base import TermCommand
from cli.constants import BM25_B, BM25_K1
from cli.schemas import BM25Payload, Request, TermWithDocIDPayload

P = TypeVar("P", bound=TermWithDocIDPayload)


class BaseComputeTFCommand(TermCommand):
    """Abstract base for TF commands; registers doc_id and term positional arguments."""

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Register doc_id and term positional arguments with the tf subparser.

        Args:
            parser (ArgumentParser): The tf subparser.
        """
        parser.add_argument("doc_id", type=int, help="Document ID")
        super().add_arguments(parser)


class ComputeTFCommand(BaseComputeTFCommand):
    """Command that loads the cached index and prints the term frequency for a doc."""

    term_help = "Term to get term frequency for"

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


class BaseTFScoreCommand(BaseComputeTFCommand, Generic[P]):
    """Abstract base for TF score commands; prints '{label} of ... : {val:.2f}'."""

    _label: str

    @override
    def run(self, request: Request[P]) -> None:
        """Load the index from cache and print the score for the given document.

        Args:
            request (Request[P]): Contains the document ID and term.
        """
        self.load_cache()
        payload = request.payload
        print(
            f"{self._label} of '{payload.term}' in document "
            f"'{payload.doc_id}': {self._score(payload):.2f}"
        )

    @abstractmethod
    def _score(self, payload: P) -> float:
        """Compute and return the score for the given payload."""


class ComputeTFIDFCommand(BaseTFScoreCommand[TermWithDocIDPayload]):
    """Command that loads the cached index and prints the TF-IDF for a term."""

    _label = "TF-IDF score"
    term_help = "Term to compute TF-IDF score for"

    @override
    def _score(self, payload: TermWithDocIDPayload) -> float:
        return self.inverted_index.get_tf(
            payload.doc_id, payload.term
        ) * self.inverted_index.get_idf(payload.term)


class ComputeBM25TFCommand(BaseTFScoreCommand[BM25Payload]):
    """Command that loads the cached index and prints the BM25 TF score for a doc."""

    _label = "BM25 TF score"
    term_help = "Term to get BM25 TF score for"

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Register doc_id, term, and optional k1 and b arguments.

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
    def _score(self, payload: BM25Payload) -> float:
        return self.inverted_index.get_bm25_tf(
            payload.doc_id,
            payload.term,
            k1=payload.k1,
            b=payload.b,
        )
