"""Verify command: loads the semantic search model and prints verification info."""

from argparse import ArgumentParser
from typing import override

from cli.commands.base import BaseCommand
from cli.schemas import EmptyPayload, Request
from cli.core.semantic_search import verify_model, verify_embeddings


class VerifyCommand(BaseCommand[EmptyPayload]):
    """Command that loads the semantic search model and prints its properties."""

    def add_arguments(self, parser: ArgumentParser) -> None:
        """No arguments are registered for the verify command.

        Args:
            parser (ArgumentParser): The verify subparser.
        """

    @override
    def run(self, request: Request[EmptyPayload]) -> None:
        """Load the semantic model and print verification details.

        Args:
            request (Request[EmptyPayload]): No payload fields required.
        """
        verify_model()


class VerifyEmbeddingsCommand(VerifyCommand):
    """Command that loads or creates corpus embeddings and prints their shape."""

    @override
    def run(self, request: Request[EmptyPayload]) -> None:
        """Load or build movie embeddings and print document count and shape.

        Args:
            request (Request[EmptyPayload]): No payload fields required.
        """
        verify_embeddings()
