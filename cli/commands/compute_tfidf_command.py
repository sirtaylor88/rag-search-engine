"""TF-IDF command: computes the TF-IDF score for a term in a document."""

from typing import override

from cli.commands.find_tf_command import FindTFCommand


class ComputeTFIDFCommand(FindTFCommand):
    """Command that loads the cached index and prints the TF-IDF for a term."""

    @override
    def run(self, doc_id: int, term: str) -> None:  # pylint: disable=arguments-differ
        """Load the index from cache and print the TF-IDF score for the given document.

        Args:
            doc_id (int): The document ID to look up.
            term (str): The term to compute TF-IDF for.
        """
        self.load_cache()

        idf = self.inverted_index.get_idf(term)
        tf = self.inverted_index.get_tf(doc_id, term)

        tf_idf = tf * idf

        print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")
