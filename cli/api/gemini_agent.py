"""Gemini API client and query-enhancement helpers."""

import logging
import os

from dotenv import load_dotenv
from google import genai

from cli.constants import DEFAULT_ENHANCE_METHOD, GEMINI_MODEL

load_dotenv()

logger = logging.getLogger(__name__)


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
    method: str = DEFAULT_ENHANCE_METHOD,
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

    prompt = f"""
    Fix any spelling errors in the user-provided movie search query below.
    Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder
    words.
    Preserve punctuation and capitalization unless a change is required for a typo fix.
    If there are no spelling errors, or if you're unsure, output the original query
    unchanged.
    Output only the final query text, nothing else.
    User query: "{query}"
    """
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
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
