"""Abstract base command and CLI parser registration."""

from abc import ABC, abstractmethod

from argparse import ArgumentParser
from typing import Any


class Command(ABC):  # pylint: disable=too-few-public-methods
    """Abstract base class for CLI commands."""

    @staticmethod
    @abstractmethod
    def run(*args: Any, **kwargs: Any) -> None:
        """Run the command."""


def register_commands() -> ArgumentParser:
    """Build and return the top-level argument parser with all subcommands registered.

    Returns:
        ArgumentParser: Parser with 'search' and 'build' subcommands.
    """
    parser = ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # * Search command
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--data-path",
        type=str,
        default="data/movies.json",
        help="Path to the movies JSON file (default: data/movies.json)",
    )

    # * Build command
    build_parser = subparsers.add_parser("build", help="Build inverted index")
    build_parser.add_argument(
        "--data-path",
        type=str,
        default="data/movies.json",
        help="Path to the movies JSON file (default: data/movies.json)",
    )

    return parser
