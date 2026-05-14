"""CLI for rewriting a text search query using an image as additional context."""

import argparse
import mimetypes

from cli.api.gemini_agent import describe_image


def main() -> None:
    """Parse CLI arguments and rewrite a query enriched with image context."""
    parser = argparse.ArgumentParser(description="Describe image CLI")
    parser.add_argument(
        "--image",
        type=str,
        help="Path to the image file",
    )
    parser.add_argument(
        "--query",
        type=str.strip,
        help="Text query to rewrite based on the image",
    )

    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(args.image, "rb") as fh:
        img = fh.read()

    gemini_response = describe_image(img, mime, args.query)
    rewritten_query = gemini_response.text.strip() if gemini_response.text else ""
    print(f"Rewritten query: {rewritten_query}")
    if gemini_response.usage_metadata:
        print(f"Total tokens:    {gemini_response.usage_metadata.total_token_count}")


if __name__ == "__main__":  # pragma: no cover
    main()
