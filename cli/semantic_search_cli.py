"""CLI for semantic movie search using sentence-transformers."""

import argparse

from cli.commands.base import EmptyPayload, EmptyRequest, TermPayload, TermRequest
from cli.commands.search import VerifyCommand, EmbedTextCommand


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate semantic search command."""
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_cmd = VerifyCommand(
        subparsers.add_parser("verify", help="Verify semantic model")
    )
    embed_text_cmd = EmbedTextCommand(
        subparsers.add_parser("embed_text", help="Verify semantic model")
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_cmd.run(EmptyRequest(payload=EmptyPayload()))
        case "embed_text":
            embed_text_cmd.run(TermRequest(payload=TermPayload(term=args.term)))
        case _:
            parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
