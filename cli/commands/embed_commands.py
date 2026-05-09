"""Embed commands: encode text or query strings into dense embeddings."""

from abc import abstractmethod
from typing import override

from cli.commands.base import TermCommand
from cli.schemas import Request, TermPayload
from cli.core.semantic_search import embed_query_text, embed_text


class BaseEmbedCommand(TermCommand):
    """Abstract base for embed commands; subclasses implement _embed."""

    @override
    def run(self, request: Request[TermPayload]) -> None:
        """Encode the term and print embedding info.

        Args:
            request (Request[TermPayload]): Payload containing the term to encode.
        """
        self._embed(request.payload.term)

    @abstractmethod
    def _embed(self, term: str) -> None:
        """Encode the term and print its info."""


class EmbedTextCommand(BaseEmbedCommand):
    """Command that encodes a text string and prints its dense embedding info."""

    @override
    def _embed(self, term: str) -> None:
        embed_text(term)


class EmbedQueryCommand(BaseEmbedCommand):
    """Command that encodes a query string and prints its dense embedding info."""

    @override
    def _embed(self, term: str) -> None:
        embed_query_text(term)
