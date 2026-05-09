"""Build command: loads movies from JSON and constructs the inverted index."""

from argparse import ArgumentParser
import sys
from typing import override

from cli.commands.base import BaseCommand
from cli.schemas import Request, TermPayload
from cli.utils import get_movies


class BuildCommand(BaseCommand[TermPayload]):
    """Command that builds and persists the inverted index from a movies JSON file."""

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Register --data-path argument with the build subparser.

        Args:
            parser (ArgumentParser): The build subparser.
        """
        parser.add_argument(
            "--data-path",
            type=str,
            dest="term",
            default="data/movies.json",
            help="Path to the movies JSON file (default: data/movies.json)",
        )

    @override
    def run(self, request: Request[TermPayload]) -> None:
        """Load movies, build the inverted index, and save it to cache.

        Args:
            request (Request[TermPayload]): Contains the path to the movies JSON file.
        """
        try:
            movies = get_movies(data_path=request.payload.term)
        except OSError:
            print("Cannot build movies data.")
            sys.exit(1)

        print("Building inverted index for", len(movies), "movies...")

        self.inverted_index.build(movies)
        self.inverted_index.save()

        print("Inverted index has been built !")
