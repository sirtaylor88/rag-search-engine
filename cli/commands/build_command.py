"""Build command: loads movies from JSON and constructs the inverted index."""

from argparse import Namespace
import json
import sys
from typing import override

from cli.inverted_index import Document, InvertedIndex
from cli.commands.base import Command


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


class BuildCommand(Command):  # pylint: disable=too-few-public-methods
    """Command that builds and persists the inverted index from a movies JSON file."""

    @override
    @staticmethod
    def run(args: Namespace, inverted_index: InvertedIndex) -> None:  # pylint: disable=arguments-differ
        """Load movies, build the inverted index, and save it to cache.

        Args:
            args (Namespace): Parsed CLI arguments; uses args.data_path.
            inverted_index (InvertedIndex): Empty index to populate and save.
        """
        try:
            movies = get_movies(data_path=args.data_path)
        except OSError:
            print("Cannot build movies data.")
            sys.exit(1)

        print("Building inverted index for", len(movies), "movies...")

        inverted_index.build(movies)
        inverted_index.save()

        print("Inverted index has been built !")
