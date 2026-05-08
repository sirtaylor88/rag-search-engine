"""Embed-text command: encodes a text string and prints its embedding info."""

from typing import override

from cli.commands.base import Request, TermCommand, TermPayload
from cli.core.semantic_search import embed_text


class EmbedTextCommand(TermCommand):
    """Command that encodes a text string and prints its dense embedding info."""

    @override
    def run(self, request: Request[TermPayload]) -> None:
        """Encode the given text and print the embedding dimensions.

        Args:
            request (Request[TermPayload]): Payload containing the text to encode.
        """
        embed_text(request.payload.term)
