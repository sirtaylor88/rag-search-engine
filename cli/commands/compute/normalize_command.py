"""Normalize command: min-max normalises a list of positive scores."""

from typing import override

from cli.commands.base import BaseListCommand
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

        min_score = min(scores)
        max_score = max(scores)

        for score in scores:
            try:
                normalized_score = (score - min_score) / (max_score - min_score)
            except ZeroDivisionError:
                normalized_score = 1.0

            print(f"* {normalized_score:.4f}")
