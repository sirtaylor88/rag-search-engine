"""TF-IDF command: computes the TF-IDF score for a term in a document."""

from typing import override

from cli.commands.find_tf_command import FindTFCommand
from cli.commands.base import TermWithDocIDRequest


class ComputeTFIDFCommand(FindTFCommand):
    """Command that loads the cached index and prints the TF-IDF for a term."""

    @override
    def run(self, request: TermWithDocIDRequest) -> None:
        """Load the index from cache and print the TF-IDF score for the given document.

        Args:
            request (TermWithDocIDRequest): Contains the document ID and term.
        """
        self.load_cache()

        idf = self.inverted_index.get_idf(request.term)
        tf = self.inverted_index.get_tf(request.doc_id, request.term)

        tf_idf = tf * idf

        print(
            f"TF-IDF score of '{request.term}' in document "
            f"'{request.doc_id}': {tf_idf:.2f}"
        )
