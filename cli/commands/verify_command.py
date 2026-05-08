"""Verify command: loads the semantic search model and prints verification info."""

from argparse import ArgumentParser
from typing import override

from cli.commands.base import BaseCommand, EmptyPayload, Request
from cli.core.semantic_search import verify_model


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
