"""Embed commands: encode text or query strings into dense embeddings."""

from typing import override

from cli.commands.base import Request, TermCommand, TermPayload
from cli.core.semantic_search import embed_query_text, embed_text


class EmbedQueryCommand(TermCommand):
    """Command that encodes a query string and prints its dense embedding info."""

    @override
    def run(self, request: Request[TermPayload]) -> None:
        """Encode the given query and print its embedding shape.

        Args:
            request (Request[TermPayload]): Payload containing the query to encode.
        """
        embed_query_text(request.payload.term)


class EmbedTextCommand(TermCommand):
    """Command that encodes a text string and prints its dense embedding info."""

    @override
    def run(self, request: Request[TermPayload]) -> None:
        """Encode the given text and print the embedding dimensions.

        Args:
            request (Request[TermPayload]): Payload containing the text to encode.
        """
        embed_text(request.payload.term)
