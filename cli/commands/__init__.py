"""CLI command classes and parser registration."""

from cli.commands.search_command import SearchCommand
from cli.commands.build_command import BuildCommand
from cli.commands.base import register_commands

__all__ = [
    # * Class
    "BuildCommand",
    "SearchCommand",
    # * Methods
    "register_commands",
]
