"""Tests for cli.core.multimodal_search."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pytest import CaptureFixture

from cli.core.keyword_search import Document
from cli.core.multimodal_search import MultimodalSearch, verify_image_embedding

DOCS: list[Document] = [
    {"id": 1, "title": "Movie A", "description": "A great film about adventure"},
    {"id": 2, "title": "Movie B", "description": "A romantic comedy"},
]


@pytest.fixture(autouse=True)
def _reset_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the MultimodalSearch singleton before each test."""
    monkeypatch.setattr(MultimodalSearch, "_instance", None)


def _make_model() -> MagicMock:
    """Build a mock SentenceTransformer returning [[0.1, 0.2, 0.3]] from encode."""
    model = MagicMock()
    model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    return model


class TestMultimodalSearch:
    """Tests for MultimodalSearch."""

    def test_embed_image_returns_ndarray(self) -> None:
        """embed_image should return a 1-D ndarray from the model output."""
        with (
            patch(
                "cli.core.multimodal_search.SentenceTransformer",
                return_value=_make_model(),
            ),
            patch("cli.core.multimodal_search.Image.open", return_value=MagicMock()),
        ):
            result = MultimodalSearch(DOCS).embed_image("img.jpg")

        assert result.shape == (3,)
        assert np.allclose(result, [0.1, 0.2, 0.3])

    def test_embed_image_opens_file_path(self) -> None:
        """embed_image should open the given file path via Image.open."""
        with (
            patch(
                "cli.core.multimodal_search.SentenceTransformer",
                return_value=_make_model(),
            ),
            patch(
                "cli.core.multimodal_search.Image.open", return_value=MagicMock()
            ) as mock_open,
        ):
            MultimodalSearch(DOCS).embed_image("poster.jpg")

        mock_open.assert_called_once_with("poster.jpg")

    def test_singleton_returns_same_instance(self) -> None:
        """Calling MultimodalSearch() twice should return the same object."""
        with patch(
            "cli.core.multimodal_search.SentenceTransformer", return_value=MagicMock()
        ):
            ms1 = MultimodalSearch(DOCS)
            ms2 = MultimodalSearch(DOCS)

        assert ms1 is ms2

    def test_search_with_image_returns_ranked_results(self) -> None:
        """search_with_image should rank documents by cosine similarity to the image."""
        mock_model = MagicMock()
        mock_model.encode.side_effect = [
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),  # text embeddings (init)
            np.array([[1.0, 0.0, 0.0]]),  # image embedding (embed_image)
        ]

        with (
            patch(
                "cli.core.multimodal_search.SentenceTransformer",
                return_value=mock_model,
            ),
            patch("cli.core.multimodal_search.Image.open", return_value=MagicMock()),
        ):
            ms = MultimodalSearch(DOCS)
            results = ms.search_with_image("img.jpg", limit=2)

        assert len(results) == 2
        assert results[0]["title"] == "Movie A"
        assert results[0]["score"] >= results[1]["score"]

    def test_search_with_image_respects_limit(self) -> None:
        """search_with_image should return at most ``limit`` results."""
        mock_model = MagicMock()
        mock_model.encode.side_effect = [
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            np.array([[1.0, 0.0, 0.0]]),
        ]

        with (
            patch(
                "cli.core.multimodal_search.SentenceTransformer",
                return_value=mock_model,
            ),
            patch("cli.core.multimodal_search.Image.open", return_value=MagicMock()),
        ):
            results = MultimodalSearch(DOCS).search_with_image("img.jpg", limit=1)

        assert len(results) == 1


class TestVerifyImageEmbedding:
    """Tests for verify_image_embedding."""

    def test_prints_embedding_shape(self, capsys: CaptureFixture[str]) -> None:
        """Should print the number of dimensions in the embedding."""
        embedding = np.zeros(512)
        with (
            patch("cli.core.multimodal_search.load_movies", return_value=DOCS),
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
