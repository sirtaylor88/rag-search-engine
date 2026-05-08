"""IDF command: computes inverse document frequency for a term."""

from typing import override

from cli.commands.base import TermCommand, TermRequest


class ComputeIDFCommand(TermCommand):
    """Command that loads the cached index and prints the IDF for a term."""

    term_help = "Term to get IDF score for"

    @override
    def run(self, request: TermRequest) -> None:
        """Load the index from cache and print the IDF for the given term.

        Args:
            request (TermRequest): Contains the term to compute IDF for.
        """
        self.load_cache()

        idf = self.inverted_index.get_idf(request.term)

        print(f"Inverse document frequency of '{request.term}': {idf:.2f}")
