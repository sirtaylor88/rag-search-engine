"""BM25 IDF command: computes Okapi BM25 inverse document frequency for a term."""

from typing import override

from cli.commands.compute_idf_command import ComputeIDFCommand
from cli.commands.base import TermRequest


class ComputeBM25IDFCommand(ComputeIDFCommand):
    """Command that loads the cached index and prints the BM25 IDF for a term."""

    term_help = "Term to get BM25 IDF score for"

    @override
    def run(self, request: TermRequest) -> None:
        """Load the index from cache and print the BM25 IDF for the given term.

        Args:
            request (TermRequest): Contains the term to compute BM25 IDF for.
        """
        self.load_cache()

        bm25idf = self.inverted_index.get_bm25_idf(request.term)

        print(f"BM25 IDF score of '{request.term}': {bm25idf:.2f}")
