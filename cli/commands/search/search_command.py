"""Search command: queries the inverted index and prints the top results."""

from typing import override

from cli.commands.base import BaseSearchCommand
from cli.schemas import Request, SearchPayload


class SearchCommand(BaseSearchCommand):
    """Command that loads the cached index and prints the top matching movies."""

    @override
    def run(self, request: Request[SearchPayload]) -> None:
        """Load the index from cache and display the best matching results.

        Args:
            request (Request[SearchPayload]): Contains the search query string.
        """
        self.load_cache()

        print("Searching for:", request.payload.query)

        results = self.inverted_index.search(
            request.payload.query, request.payload.limit
        )

        for doc_id in results:
            title = self.inverted_index.docmap[doc_id]["title"]
            print(f"{doc_id}. {title}")
