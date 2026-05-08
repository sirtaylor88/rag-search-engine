"""CLI command classes."""

from cli.commands.search_command import SearchCommand
from cli.commands.build_command import BuildCommand
from cli.commands.compute.compute_tf_command import ComputeTFCommand
from cli.commands.compute.compute_idf_command import ComputeIDFCommand
from cli.commands.compute.compute_tfidf_command import ComputeTFIDFCommand
from cli.commands.compute.compute_bm25_idf_command import ComputeBM25IDFCommand
from cli.commands.compute.compute_bm25_tf_command import ComputeBM25TFCommand

__all__ = [
    "BuildCommand",
    "ComputeBM25IDFCommand",
    "ComputeBM25TFCommand",
    "ComputeIDFCommand",
    "ComputeTFIDFCommand",
    "ComputeTFCommand",
    "SearchCommand",
]
