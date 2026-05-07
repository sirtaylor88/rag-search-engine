"""Build command: loads movies from JSON and constructs the inverted index."""

from argparse import ArgumentParser
import json
import sys
from typing import override

from cli.inverted_index import Document
from cli.commands.base import BaseCommand


def get_movies(data_path: str = "data/movies.json") -> list[Document]:
    """Load and return the list of movies from a JSON file.

    Args:
        data_path (str): Path to the JSON file containing movie data.

    Returns:
        list[Document]: List of movie documents from the 'movies' key.
    """
    with open(data_path, encoding="utf-8") as fh:
        data = json.load(fh)

    return data["movies"]


class BuildCommand(BaseCommand):
    """Command that builds and persists the inverted index from a movies JSON file."""

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Register --data-path argument with the build subparser.

        Args:
            parser (ArgumentParser): The build subparser.
        """
        parser.add_argument(
            "--data-path",
            type=str,
            default="data/movies.json",
            help="Path to the movies JSON file (default: data/movies.json)",
        )

    @override
    def run(self, data_path: str) -> None:  # pylint: disable=arguments-differ
        """Load movies, build the inverted index, and save it to cache.

        Args:
            data_path (str): Path to the movies JSON file.
        """
        try:
            movies = get_movies(data_path=data_path)
        except OSError:
            print("Cannot build movies data.")
            sys.exit(1)

        print("Building inverted index for", len(movies), "movies...")

        self.inverted_index.build(movies)
        self.inverted_index.save()

        print("Inverted index has been built !")
