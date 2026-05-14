"""CLI for multimodal image-embedding verification."""

import argparse

from cli.commands import VerifyImageEmbeddingCommand
from cli.schemas.payloads import TermPayload
from cli.schemas.requests import TermRequest


def main() -> None:
    """Parse CLI arguments and dispatch to the multimodal search command."""
    parser = argparse.ArgumentParser(description="Multimodal search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_cmd = VerifyImageEmbeddingCommand(
        subparsers.add_parser("verify_image_embedding", help="Verify image embedding."),
    )

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding_cmd.run(
                TermRequest(payload=TermPayload(term=args.term))
            )

        case _:
            parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
