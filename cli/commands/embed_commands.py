"""Embed commands: encode text or query strings into dense embeddings."""

from abc import abstractmethod
from argparse import ArgumentParser
from typing import override

from cli.commands.base import BaseCommand, TermCommand
from cli.core.semantic_search import embed_chunks, embed_query_text, embed_text
from cli.schemas import EmptyPayload, Request, TermPayload


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


class EmbedChunksCommand(BaseCommand[EmptyPayload]):
    """Command that encodes all corpus sentence chunks and prints the total count."""

    def add_arguments(self, parser: ArgumentParser) -> None:
        """No arguments registered for the embed_chunks command.

        Args:
            parser (ArgumentParser): The embed_chunks subparser.
        """

    @override
    def run(self, request: Request[EmptyPayload]) -> None:
        """Load or create chunk embeddings for the corpus and print the count.

        Args:
            request (Request[EmptyPayload]): No payload fields required.
        """
        embed_chunks()
