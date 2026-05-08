"""CLI command classes."""

from cli.commands.search_command import SearchCommand
from cli.commands.build_command import BuildCommand
from cli.commands.term_frequency_command import TermFrequecyCommand

__all__ = ["BuildCommand", "SearchCommand", "TermFrequecyCommand"]
