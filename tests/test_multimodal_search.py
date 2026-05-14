"""Tests for cli.core.multimodal_search."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pytest import CaptureFixture

from cli.core.multimodal_search import MultimodalSearch, verify_image_embedding


@pytest.fixture(autouse=True)
def _reset_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the MultimodalSearch singleton before each test."""
    monkeypatch.setattr(MultimodalSearch, "_instance", None)


def _make_model(embedding: list[float] | None = None) -> MagicMock:
    """Build a mock SentenceTransformer that returns a fixed embedding."""
    model = MagicMock()
    model.encode.return_value = np.array([embedding or [0.1, 0.2, 0.3]])
    return model


class TestMultimodalSearch:
    """Tests for MultimodalSearch."""

    def test_embed_image_returns_ndarray(self) -> None:
        """embed_image should return a 1-D ndarray from the model output."""
        mock_model = _make_model([0.1, 0.2, 0.3])
        with (
            patch(
                "cli.core.multimodal_search.SentenceTransformer",
                return_value=mock_model,
            ),
            patch("cli.core.multimodal_search.Image.open", return_value=MagicMock()),
        ):
            result = MultimodalSearch().embed_image("img.jpg")

        assert result.shape == (3,)
        assert np.allclose(result, [0.1, 0.2, 0.3])

    def test_embed_image_opens_file_path(self) -> None:
        """embed_image should open the given file path via Image.open."""
        mock_model = _make_model()
        with (
            patch(
                "cli.core.multimodal_search.SentenceTransformer",
                return_value=mock_model,
            ),
            patch(
                "cli.core.multimodal_search.Image.open", return_value=MagicMock()
            ) as mock_open,
        ):
            MultimodalSearch().embed_image("poster.jpg")

        mock_open.assert_called_once_with("poster.jpg")

    def test_singleton_returns_same_instance(self) -> None:
        """Calling MultimodalSearch() twice should return the same object."""
        with patch(
            "cli.core.multimodal_search.SentenceTransformer", return_value=MagicMock()
        ):
            ms1 = MultimodalSearch()
            ms2 = MultimodalSearch()

        assert ms1 is ms2


class TestVerifyImageEmbedding:
    """Tests for verify_image_embedding."""

    def test_prints_embedding_shape(self, capsys: CaptureFixture[str]) -> None:
        """Should print the number of dimensions in the embedding."""
        embedding = np.zeros(512)
        with (
            patch(
                "cli.core.multimodal_search.SentenceTransformer",
                return_value=MagicMock(),
            ),
            patch.object(MultimodalSearch, "embed_image", return_value=embedding),
        ):
            verify_image_embedding("img.jpg")

        out = capsys.readouterr().out
        assert "512" in out
        assert "Embedding shape" in out
