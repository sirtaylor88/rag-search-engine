"""Tests for cli.core.semantic_search."""

from unittest.mock import MagicMock, patch

import pytest
from pytest import CaptureFixture

from cli.core.semantic_search import SemanticSearch, embed_text, verify_model


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

    def test_generate_embedding_returns_encoded_tensor(self) -> None:
        """generate_embedding should call model.encode and return the first element."""
        mock_tensor = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = [mock_tensor]
        with patch(
            "cli.core.semantic_search.SentenceTransformer", return_value=mock_model
        ):
            ss = SemanticSearch()
            result = ss.generate_embedding("hello world")

        mock_model.encode.assert_called_once_with(["hello world"])
        assert result is mock_tensor

    def test_generate_embedding_raises_on_empty_text(self) -> None:
        """generate_embedding should raise ValueError for whitespace-only text."""
        mock_model = MagicMock()
        with patch(
            "cli.core.semantic_search.SentenceTransformer", return_value=mock_model
        ):
            ss = SemanticSearch()
            with pytest.raises(ValueError, match="empty"):
                ss.generate_embedding("   ")


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


class TestEmbedText:
    """Tests for the embed_text function."""

    def test_prints_embedding_info(self, capsys: CaptureFixture[str]) -> None:
        """embed_text should print text, first 3 dimensions, and total dimensions."""
        mock_embedding = MagicMock()
        mock_embedding.__getitem__ = MagicMock(return_value="[0.1, 0.2, 0.3]")
        mock_embedding.shape = [384]
        mock_model = MagicMock()
        mock_model.encode.return_value = [mock_embedding]
        with patch(
            "cli.core.semantic_search.SentenceTransformer", return_value=mock_model
        ):
            embed_text("hello")

        out = capsys.readouterr().out
        assert "Text: hello" in out
        assert "First 3 dimensions:" in out
        assert "Dimensions:" in out
