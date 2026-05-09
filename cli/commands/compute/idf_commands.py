"""IDF commands: compute inverse document frequency and BM25 IDF for a term."""

from typing import override

from cli.commands.base import Request, TermCommand, TermPayload


class ComputeIDFCommand(TermCommand):
    """Command that loads the cached index and prints the IDF for a term."""

    term_help = "Term to get IDF score for"

    @override
    def run(self, request: Request[TermPayload]) -> None:
        """Load the index from cache and print the IDF for the given term.

        Args:
            request (Request[TermPayload]): Contains the term to compute IDF for.
        """
        self.load_cache()

        idf = self.inverted_index.get_idf(request.payload.term)

        print(f"Inverse document frequency of '{request.payload.term}': {idf:.2f}")


class ComputeBM25IDFCommand(ComputeIDFCommand):
    """Command that loads the cached index and prints the BM25 IDF for a term."""

    term_help = "Term to get BM25 IDF score for"

    @override
    def run(self, request: Request[TermPayload]) -> None:
        """Load the index from cache and print the BM25 IDF for the given term.

        Args:
            request (Request[TermPayload]): Contains the term to compute BM25 IDF for.
        """
        self.load_cache()

        bm25idf = self.inverted_index.get_bm25_idf(request.payload.term)

        print(f"BM25 IDF score of '{request.payload.term}': {bm25idf:.2f}")
