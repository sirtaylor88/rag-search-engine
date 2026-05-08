"""TF-IDF command: computes the TF-IDF score for a term in a document."""

from typing import override

from cli.commands.compute.compute_tf_command import ComputeTFCommand
from cli.commands.base import Request, TermWithDocIDPayload


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
