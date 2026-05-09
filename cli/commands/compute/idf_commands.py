"""IDF commands: compute inverse document frequency and BM25 IDF for a term."""

from abc import abstractmethod
from typing import override

from cli.commands.base import TermCommand
from cli.schemas import Request, TermPayload


class BaseComputeIDFCommand(TermCommand):
    """Abstract base for IDF commands; loads cache and prints the IDF score."""

    _label: str

    @override
    def run(self, request: Request[TermPayload]) -> None:
        """Load the index from cache and print the IDF score for the given term.

        Args:
            request (Request[TermPayload]): Contains the term to compute IDF for.
        """
        self.load_cache()
        term = request.payload.term
        print(f"{self._label} of '{term}': {self._compute_idf(term):.2f}")

    @abstractmethod
    def _compute_idf(self, term: str) -> float:
        """Return the IDF score for the term."""


class ComputeIDFCommand(BaseComputeIDFCommand):
    """Command that loads the cached index and prints the smoothed IDF for a term."""

    _label = "Inverse document frequency"
    term_help = "Term to get IDF score for"

    @override
    def _compute_idf(self, term: str) -> float:
        return self.inverted_index.get_idf(term)


class ComputeBM25IDFCommand(BaseComputeIDFCommand):
    """Command that loads the cached index and prints the BM25 IDF for a term."""

    _label = "BM25 IDF score"
    term_help = "Term to get BM25 IDF score for"

    @override
    def _compute_idf(self, term: str) -> float:
        return self.inverted_index.get_bm25_idf(term)
