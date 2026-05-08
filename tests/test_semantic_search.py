"""Tests for cli.core.semantic_search."""

from unittest.mock import MagicMock, patch

from pytest import CaptureFixture

from cli.core.semantic_search import SemanticSearch, verify_model


class TestSemanticSearch:
    """Tests for the SemanticSearch class."""

    def test_loads_model_on_init(self) -> None:
        """SemanticSearch should load the all-MiniLM-L6-v2 model on instantiation."""
        mock_model = MagicMock()
        with patch(
            "cli.core.semantic_search.SentenceTransformer", return_value=mock_model
        ) as mock_st:
            ss = SemanticSearch()
            mock_st.assert_called_once_with("all-MiniLM-L6-v2")
            assert ss.model is mock_model


class TestVerifyModel:
    """Tests for the verify_model function."""

    def test_prints_model_and_sequence_length(
        self, capsys: CaptureFixture[str]
    ) -> None:
        """verify_model should print the model and its max_seq_length."""
        mock_model = MagicMock()
        mock_model.max_seq_length = 128
        with patch(
            "cli.core.semantic_search.SentenceTransformer", return_value=mock_model
        ):
            verify_model()

        out = capsys.readouterr().out
        assert "Model loaded:" in out
        assert "Max sequence length:" in out
