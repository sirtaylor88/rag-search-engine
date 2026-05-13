"""Tests for cli.api.gemini_agent."""

from unittest.mock import MagicMock, patch

import pytest

from cli.api.gemini_agent import (
    augment_query,
    enhance_query,
    evaluate_query,
    get_gemini_client,
    rerank_query,
)


def _make_response(text: str | None) -> MagicMock:
    """Build a mock Gemini response."""
    response = MagicMock()
    response.text = text
    response.usage_metadata.prompt_token_count = 10
    response.usage_metadata.candidates_token_count = 5
    return response


class TestGetGeminiClient:
    """Tests for get_gemini_client."""

    def test_raises_when_api_key_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should raise RuntimeError when GEMINI_API_KEY is absent."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            get_gemini_client()

    def test_returns_client_when_api_key_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should instantiate and return a genai.Client with the API key."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        mock_client = MagicMock()
        with patch(
            "cli.api.gemini_agent.genai.Client", return_value=mock_client
        ) as mock_ctor:
            result = get_gemini_client()

        mock_ctor.assert_called_once_with(api_key="test-key")
        assert result is mock_client


class TestEnhanceQuery:
    """Tests for enhance_query."""

    def test_returns_original_when_method_is_none(self) -> None:
        """Should return the original query immediately without creating a client."""
        with patch("cli.api.gemini_agent.get_gemini_client") as mock_get_client:
            result = enhance_query("action movie")

        mock_get_client.assert_not_called()
        assert result == "action movie"

    def test_returns_enhanced_text_when_response_available(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Should return response.text and print the enhancement line."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_response(
            "Jurassic Park"
        )
        with patch("cli.api.gemini_agent.get_gemini_client", return_value=mock_client):
            result = enhance_query("Jurrasic Park", method="spell")

        assert result == "Jurassic Park"
        out = capsys.readouterr().out
        assert "Jurrasic Park" in out
        assert "Jurassic Park" in out

    def test_returns_original_when_no_response_text(self) -> None:
        """Should return the original query when response.text is empty."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_response(None)
        with patch("cli.api.gemini_agent.get_gemini_client", return_value=mock_client):
            result = enhance_query("action movie", method="spell")

        assert result == "action movie"

    def test_rewrite_method_returns_enhanced_text(self) -> None:
        """Should call generate_content and return its text when method='rewrite'."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_response(
            "The Revenant bear attack"
        )
        with patch("cli.api.gemini_agent.get_gemini_client", return_value=mock_client):
            result = enhance_query("bear leo movie", method="rewrite")

        assert result == "The Revenant bear attack"

    def test_raises_for_invalid_method(self) -> None:
        """Should raise ValueError for an unrecognised enhancement method."""
        mock_client = MagicMock()
        with (
            patch("cli.api.gemini_agent.get_gemini_client", return_value=mock_client),
            pytest.raises(ValueError, match="Invalid enhance method"),
        ):
            enhance_query("action", method="unknown")  # type: ignore[arg-type]

    def test_logs_token_counts(self, caplog: pytest.LogCaptureFixture) -> None:
        """Should log prompt and response token counts at INFO level."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_response("action")
        with (
            patch("cli.api.gemini_agent.get_gemini_client", return_value=mock_client),
            caplog.at_level("INFO", logger="cli.api.gemini_agent"),
        ):
            enhance_query("action", method="spell")

        messages = [r.message for r in caplog.records]
        assert any("10" in m for m in messages)
        assert any("5" in m for m in messages)

    def test_uses_method_label_in_output(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Should include the method label in the printed enhancement line."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_response("fixed")
        with patch("cli.api.gemini_agent.get_gemini_client", return_value=mock_client):
            enhance_query("typo", method="spell")

        assert "spell" in capsys.readouterr().out

    def test_expand_method_returns_enhanced_text(self) -> None:
        """Should call generate_content and return its text when method='expand'."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_response(
            "scary horror grizzly bear terrifying film"
        )
        with patch("cli.api.gemini_agent.get_gemini_client", return_value=mock_client):
            result = enhance_query("scary bear movie", method="expand")

        assert result == "scary horror grizzly bear terrifying film"


class TestReRankQuery:
    """Tests for rerank_query."""

    def test_returns_none_when_method_is_none(self) -> None:
        """Should return None immediately without creating a client."""
        with patch("cli.api.gemini_agent.get_gemini_client") as mock_get_client:
            result = rerank_query("action", "Movie A - description")

        mock_get_client.assert_not_called()
        assert result is None

    def test_returns_text_when_response_available(self) -> None:
        """Should return raw response text when the model returns a score."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_response("7.5")
        with patch("cli.api.gemini_agent.get_gemini_client", return_value=mock_client):
            result = rerank_query(
                "action", "Movie A - description", method="individual"
            )

        assert result == "7.5"

    def test_returns_none_when_no_response_text(self) -> None:
        """Should return None when the model returns no text."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_response(None)
        with patch("cli.api.gemini_agent.get_gemini_client", return_value=mock_client):
            result = rerank_query(
                "action", "Movie A - description", method="individual"
            )

        assert result is None

    def test_batch_method_returns_text(self) -> None:
        """Should return raw response text when method='batch'."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_response("[1, 2, 3]")
        with patch("cli.api.gemini_agent.get_gemini_client", return_value=mock_client):
            result = rerank_query("action", "1 - Movie A\n2 - Movie B", method="batch")

        assert result == "[1, 2, 3]"

    def test_raises_for_invalid_method(self) -> None:
        """Should raise ValueError for an unrecognised re-rank method."""
        mock_client = MagicMock()
        with (
            patch("cli.api.gemini_agent.get_gemini_client", return_value=mock_client),
            pytest.raises(ValueError, match="Invalid re-rank method"),
        ):
            rerank_query("action", "doc", method="unknown")

    def test_logs_token_counts(self, caplog: pytest.LogCaptureFixture) -> None:
        """Should log prompt and response token counts at INFO level."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_response("8")
        with (
            patch("cli.api.gemini_agent.get_gemini_client", return_value=mock_client),
            caplog.at_level("INFO", logger="cli.api.gemini_agent"),
        ):
            rerank_query("action", "Movie A - description", method="individual")

        messages = [r.message for r in caplog.records]
        assert any("10" in m for m in messages)
        assert any("5" in m for m in messages)


class TestEvaluateQuery:
    """Tests for evaluate_query."""

    def test_returns_json_scores_from_model(self) -> None:
        """Should return the raw JSON score list from the model response."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_response("[3, 1, 0]")
        with patch("cli.api.gemini_agent.get_gemini_client", return_value=mock_client):
            result = evaluate_query(
                "bear movie", ["1. Ted\n   ...", "2. Revenant\n   ..."]
            )

        assert result == "[3, 1, 0]"

    def test_returns_none_when_no_response_text(self) -> None:
        """Should return None when the model returns no text."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_response(None)
        with patch("cli.api.gemini_agent.get_gemini_client", return_value=mock_client):
            result = evaluate_query("bear movie", ["1. Ted\n   ..."])

        assert result is None

    def test_logs_token_counts(self, caplog: pytest.LogCaptureFixture) -> None:
        """Should log prompt and response token counts at INFO level."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_response("[2]")
        with (
            patch("cli.api.gemini_agent.get_gemini_client", return_value=mock_client),
            caplog.at_level("INFO", logger="cli.api.gemini_agent"),
        ):
            evaluate_query("action", ["1. Movie A\n   ..."])

        messages = [r.message for r in caplog.records]
        assert any("10" in m for m in messages)
        assert any("5" in m for m in messages)

    def test_joins_results_into_prompt(self) -> None:
        """Should join multiple result strings with newlines in the prompt."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_response("[1, 2]")
        with patch("cli.api.gemini_agent.get_gemini_client", return_value=mock_client):
            evaluate_query("action", ["result A", "result B"])

        call_kwargs = mock_client.models.generate_content.call_args
        prompt = call_kwargs[1]["contents"]
        assert "result A\nresult B" in prompt


class TestAugmentQuery:
    """Tests for augment_query."""

    def test_returns_generated_answer(self) -> None:
        """Should return the model response text as the generated answer."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_response(
            "Paddington is a 2014 film about a bear."
        )
        with patch("cli.api.gemini_agent.get_gemini_client", return_value=mock_client):
            result = augment_query("bear movie london", ["1. Paddington\n   ..."])

        assert result == "Paddington is a 2014 film about a bear."

    def test_returns_none_when_no_response_text(self) -> None:
        """Should return None when the model returns no text."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_response(None)
        with patch("cli.api.gemini_agent.get_gemini_client", return_value=mock_client):
            result = augment_query("bear movie", ["1. Ted\n   ..."])

        assert result is None

    def test_logs_token_counts(self, caplog: pytest.LogCaptureFixture) -> None:
        """Should log prompt and response token counts at INFO level."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_response("answer")
        with (
            patch("cli.api.gemini_agent.get_gemini_client", return_value=mock_client),
            caplog.at_level("INFO", logger="cli.api.gemini_agent"),
        ):
            augment_query("action", ["1. Movie A\n   ..."])

        messages = [r.message for r in caplog.records]
        assert any("10" in m for m in messages)
        assert any("5" in m for m in messages)

    def test_joins_results_into_doc_input(self) -> None:
        """Should join multiple result strings with newlines as doc_input."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _make_response("answer")
        with patch("cli.api.gemini_agent.get_gemini_client", return_value=mock_client):
            augment_query("action", ["result A", "result B"])

        call_kwargs = mock_client.models.generate_content.call_args
        prompt = call_kwargs[1]["contents"]
        assert "result A\nresult B" in prompt
