"""Abstract base command."""

from abc import ABC, abstractmethod

from argparse import ArgumentParser
import sys
from typing import Any

from cli.inverted_index import InvertedIndex


class BaseCommand(ABC):
    """Abstract base class for CLI commands."""

    def __init__(self, parser: ArgumentParser, inverted_index: InvertedIndex) -> None:
        """Register command arguments and store the shared index.

        Args:
            parser (ArgumentParser): The subparser for this command.
            inverted_index (InvertedIndex): The shared index instance.
        """
        self.inverted_index = inverted_index
        self.add_arguments(parser)

    @abstractmethod
    def add_arguments(self, parser: ArgumentParser) -> None:
        """Register the command's CLI arguments with its subparser.

        Args:
            parser (ArgumentParser): The subparser for this command.
        """

    @abstractmethod
    def run(self, *args: Any) -> None:
        """Execute the command with the given parsed argument values."""

    def load_cache(self) -> None:
        """Load the inverted index from cache, exiting with code 1 on failure."""
        try:
            self.inverted_index.load()
        except OSError:
            print("Cannot load cache data. Please run build command first.")
            sys.exit(1)
