"""BM25 search command: ranks and displays movies by full BM25 score."""

from typing import override

from cli.commands.base import SearchRequest
from cli.commands.search.search_command import SearchCommand


class BM25SearchCommand(SearchCommand):
    """Command that loads the cached index and prints the top matching movies."""

    @override
    def run(self, request: SearchRequest) -> None:
        """Load the index from cache and display the best matching results.

        Args:
            request (SearchRequest): Contains the search query string.
        """

        self.load_cache()

        print("Searching for:", request.query)

        results = self.inverted_index.bm25_search(request.query, request.limit)

        for idx, (doc_id, score) in enumerate(results, start=1):
            title = self.inverted_index.docmap[doc_id]["title"]
            print(f"{idx}. ({doc_id}) {title} - Score: {score:.2f}")
