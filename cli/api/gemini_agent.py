"""Gemini API client and query-enhancement helpers."""

from enum import StrEnum
import logging
import os
from textwrap import dedent
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from cli.constants import GEMINI_MODEL
from cli.schemas.payloads import EnhanceMethod

load_dotenv()

logger = logging.getLogger(__name__)


class EnhancePromptPattern(StrEnum):
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

    EXPAND = """Expand the user-provided movie search query below with related terms.

    Add synonyms and related concepts that might appear in movie descriptions.
    Keep expansions relevant and focused.
    Output only the additional terms; they will be appended to the original query.

    Examples:
    - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
    - "action movie with bear" -> "action thriller bear chase fight adventure"
    - "comedy with bear" -> "comedy funny bear humor lighthearted"
    """


class ReRankPromptPattern(StrEnum):
    """Prompt templates for each re-ranking method."""

    INDIVIDUAL = """Rate how well this movie matches the search query.

    Query: "{query}"
    Movie: {doc_input}

    Consider:
    - Direct relevance to query
    - User intent (what they're looking for)
    - Content appropriateness

    Rate 0-10 (10 = perfect match).
    Output ONLY the number in your response, no other text or explanation.

    Score:"""

    BATCH = """Rank the movies listed below by relevance to the following search query.

    Query: "{query}"

    Movies:
    {doc_input}

    Return ONLY the movie IDs in order of relevance (best match first).
    Return a valid JSON list, nothing else.

    For example:
    [75, 12, 34, 2, 1]

    Ranking:"""

    EVALUATE = """Rate how relevant each result is to this query on a 0-3 scale:

    Query: "{query}"

    Results:
    {doc_input}

    Scale:
    - 3: Highly relevant
    - 2: Relevant
    - 1: Marginally relevant
    - 0: Not relevant

    Do NOT give any numbers other than 0, 1, 2, or 3.

    Return ONLY the scores in the same order you were given the documents.
    Return a valid JSON list, nothing else. For example:

    [2, 0, 3, 2, 0, 1]
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
    """Enhance a search query using a Gemini language model.

    Sends the query to Gemini using the prompt template selected by ``method``.
    Returns the original query unchanged when ``method`` is ``None`` or the
    model returns no text.

    Args:
        query (str): The original search query to enhance.
        method (EnhanceMethod, optional): Enhancement method — ``"spell"``,
            ``"rewrite"``, or ``"expand"``. Defaults to ``None`` (no-op).

    Returns:
        str: The enhanced query, or the original if no changes were made.
    """
    if method is None:
        return query

    client = get_gemini_client()

    try:
        prompt_pattern = EnhancePromptPattern[method.upper()]
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


def rerank_query(
    query: str,
    doc_input: str,
    method: Optional[str] = None,
) -> Optional[str]:
    """Score or rank documents against a query using a Gemini language model.

    Sends the query and document(s) to Gemini using the prompt template
    selected by ``method``. Returns ``None`` when ``method`` is ``None``
    or the model returns no text.

    Args:
        query (str): The search query.
        doc_input (str): The document text (or batch of documents) to score.
        method (str, optional): Re-ranking method — ``"individual"`` or
            ``"batch"``. Defaults to ``None`` (returns ``None`` immediately).

    Returns:
        Optional[str]: The raw model response text, or ``None`` if ``method``
            is ``None`` or the model returns nothing.
    """
    if method is None:
        return None

    client = get_gemini_client()

    try:
        prompt_pattern = ReRankPromptPattern[method.upper()]
    except KeyError as err:
        raise ValueError(f"Invalid re-rank method ``{method}``") from err

    prompt = dedent(prompt_pattern.value).format(
        query=query,
        doc_input=doc_input,
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                ),
            ]
        ),
    )

    if response.usage_metadata:
        logger.info("Prompt tokens: %d", response.usage_metadata.prompt_token_count)
        logger.info(
            "Response tokens: %d", response.usage_metadata.candidates_token_count
        )

    return response.text


def evaluate_query(
    query: str,
    results: list[str],
) -> Optional[str]:
    """Rate the relevance of retrieved results to a query using a Gemini model.

    Sends the query and formatted result strings to Gemini and asks it to score
    each result on a 0–3 scale. Returns a JSON list of integer scores in the
    same order as ``results``, or ``None`` if the model returns no text.

    Args:
        query (str): The original search query.
        results (list[str]): Formatted result strings (one per retrieved doc).

    Returns:
        Optional[str]: A JSON list of integer scores (``"[2, 0, 3, ...]"``),
            or ``None`` if the model returns nothing.
    """
    client = get_gemini_client()
    prompt = dedent(ReRankPromptPattern.EVALUATE.value).format(
        query=query,
        doc_input="\n".join(results),
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    if response.usage_metadata:
        logger.info("Prompt tokens: %d", response.usage_metadata.prompt_token_count)
        logger.info(
            "Response tokens: %d", response.usage_metadata.candidates_token_count
        )

    return response.text
