"""Normalize command: min-max normalises a list of positive scores."""

from typing import override

from cli.commands.base import BaseListCommand
from cli.core.hybrid_search import normalize_scores
from cli.schemas import Request
from cli.schemas.payloads import ScoreListPayload


class NormalizeCommand(BaseListCommand):
    """Command that applies min-max normalisation to a list of positive scores."""

    args_type = float
    args_help = "A list of positive scores"

    @override
    def run(self, request: Request[ScoreListPayload]) -> None:
        """Print each score normalised to [0, 1] using min-max scaling.

        Args:
            request (Request[ScoreListPayload]): Contains the list of scores.
        """
        scores = request.payload.scores

        if not scores:
            return

        results = normalize_scores(scores)

        for score in results:
            print(f"* {score:.4f}")
