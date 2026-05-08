"""IDF command: computes inverse document frequency for a term."""

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
