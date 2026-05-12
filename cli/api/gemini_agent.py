"""Gemini API client and query-enhancement helpers."""

from enum import StrEnum
import logging
import os
from textwrap import dedent
from typing import Optional

from dotenv import load_dotenv
from google import genai

from cli.constants import GEMINI_MODEL
from cli.schemas.payloads import EnhanceMethod

load_dotenv()

logger = logging.getLogger(__name__)


class PromptPattern(StrEnum):
    """Prompt templates for each query enhancement method."""

    SPELL = """Fix any spelling errors in the user-provided movie search query below.
    Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder
    words.
    Preserve punctuation and capitalization unless a change is required for a typo fix.
    If there are no spelling errors, or if you're unsure, output the original query
    unchanged.
    Output only the final query text, nothing else.
    """

    REWRITE = """Rewrite the user-provided movie search query below to be more specific
    and searchable.

    Consider:
    - Common movie knowledge (famous actors, popular films)
    - Genre conventions (horror = scary, animation = cartoon)
    - Keep the rewritten query concise (under 10 words)
    - It should be a Google-style search query, specific enough to yield relevant results
    - Don't use boolean logic

    Examples:
    - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio
    bear attack"
    - "movie about bear in london with marmalade" -> "Paddington London marmalade"
    - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

    If you cannot improve the query, output the original unchanged.
    Output only the rewritten query text, nothing else.
    """


def get_gemini_client() -> genai.Client:
    """Create a Gemini API client authenticated with GEMINI_API_KEY.

    Returns:
        genai.Client: An authenticated Gemini client.

    Raises:
        RuntimeError: If the ``GEMINI_API_KEY`` environment variable is not set.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    return genai.Client(api_key=api_key)


def enhance_query(
    query: str,
    method: Optional[EnhanceMethod] = None,
) -> str:
    """Fix spelling errors in a search query using the Gemini language model.

    Sends the query to Gemini with instructions to correct only high-confidence
    typos, leaving the rest of the text unchanged. If the model returns no text
    the original query is returned as-is.

    Args:
        query (str): The original search query to enhance.
        method (str): Enhancement method label used for display only.

    Returns:
        str: The corrected query, or the original if no changes were needed.
    """
    client = get_gemini_client()

    if method is None:
        return query

    try:
        prompt_pattern = PromptPattern[method.upper()]
    except KeyError as err:
        raise ValueError(f"Invalid enhance method ``{method}``") from err

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f'{dedent(prompt_pattern.value)}\nUser query: "{query}"',
    )

    enhanced_query = query
    if response.text:
        enhanced_query = response.text
        print(f"Enhanced query ({method}): '{query}' -> '{enhanced_query}'\n")

    if response.usage_metadata:
        logger.info("Prompt tokens: %d", response.usage_metadata.prompt_token_count)
        logger.info(
            "Response tokens: %d", response.usage_metadata.candidates_token_count
        )

    return enhanced_query
