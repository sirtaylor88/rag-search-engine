"""CLI command classes."""

from cli.commands.search_command import SearchCommand
from cli.commands.build_command import BuildCommand
from cli.commands.find_tf_command import FindTFCommand
from cli.commands.compute_idf_command import ComputeIDFCommand
from cli.commands.compute_tfidf_command import ComputeTFIDFCommand
from cli.commands.compute_bm25_idf_command import ComputeBM25IDFCommand

__all__ = [
    "BuildCommand",
    "ComputeBM25IDFCommand",
    "ComputeIDFCommand",
    "ComputeTFIDFCommand",
    "FindTFCommand",
    "SearchCommand",
]
