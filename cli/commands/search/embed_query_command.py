"""Embed-query command: encodes a query string and prints its embedding info."""

from typing import override

from cli.commands.base import Request, TermCommand, TermPayload
from cli.core.semantic_search import embed_query_text


class EmbedQueryCommand(TermCommand):
    """Command that encodes a query string and prints its dense embedding info."""

    @override
    def run(self, request: Request[TermPayload]) -> None:
        """Encode the given query and print its embedding shape.

        Args:
            request (Request[TermPayload]): Payload containing the query to encode.
        """
        embed_query_text(request.payload.term)
