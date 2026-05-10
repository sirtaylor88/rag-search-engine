"""Tests for cli.core.semantic_search."""

import json
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest
from pytest import CaptureFixture

from cli.constants import DEFAULT_EMBEDDING_MODEL
from cli.core.semantic_search import (
    ChunkedSemanticSearch,
    SemanticSearch,
    embed_chunks,
    embed_query_text,
    embed_text,
    verify_embeddings,
    verify_model,
)


@pytest.fixture(autouse=True)
def _reset_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset SemanticSearch and ChunkedSemanticSearch singletons before each test."""
    monkeypatch.setattr(SemanticSearch, "_instance", None)
    monkeypatch.setattr(ChunkedSemanticSearch, "_instance", None)


class TestSemanticSearch:
    """Tests for the SemanticSearch class."""

    def test_returns_same_instance_on_multiple_calls(self) -> None:
        """Two SemanticSearch() calls should return the identical object."""
        mock_model = MagicMock()
        with patch(
            "cli.core.semantic_search.SentenceTransformer", return_value=mock_model
        ):
            assert SemanticSearch() is SemanticSearch()

    def test_loads_model_on_init(self) -> None:
        """SemanticSearch should load the all-MiniLM-L6-v2 model on instantiation."""
        mock_model = MagicMock()
        with patch(
            "cli.core.semantic_search.SentenceTransformer", return_value=mock_model
        ) as mock_st:
            ss = SemanticSearch()
            mock_st.assert_called_once_with(DEFAULT_EMBEDDING_MODEL)
            assert ss.model is mock_model

    def test_generate_embedding_returns_encoded_tensor(self) -> None:
        """generate_embedding should call model.encode and return the first element."""
        mock_embedding = np.array([1.0, 2.0, 3.0])
        mock_model = MagicMock()
        mock_model.encode.return_value = [mock_embedding]
        with patch(
            "cli.core.semantic_search.SentenceTransformer", return_value=mock_model
        ):
            ss = SemanticSearch()
            result = ss.generate_embedding("hello world")

        mock_model.encode.assert_called_once_with(["hello world"])
        assert np.array_equal(result, mock_embedding)

    def test_generate_embedding_raises_on_empty_text(self) -> None:
        """generate_embedding should raise ValueError for whitespace-only text."""
        mock_model = MagicMock()
        with patch(
            "cli.core.semantic_search.SentenceTransformer", return_value=mock_model
        ):
            ss = SemanticSearch()
            with pytest.raises(ValueError, match="empty"):
                ss.generate_embedding("   ")

    def test_build_embeddings_populates_documents_and_map(self) -> None:
        """build_embeddings should set documents and document_map via _populate_docs."""
        docs = [
            {"id": 1, "title": "A", "description": "desc A"},
            {"id": 2, "title": "B", "description": "desc B"},
        ]
        mock_model = MagicMock()
        with (
            patch(
                "cli.core.semantic_search.SentenceTransformer",
                return_value=mock_model,
            ),
            patch("cli.core.semantic_search.np.save"),
        ):
            ss = SemanticSearch()
            ss.build_embeddings(docs)  # type: ignore[arg-type]

        assert ss.documents == docs
        assert ss.document_map == {1: docs[0], 2: docs[1]}

    def test_build_embeddings_encodes_and_saves(self) -> None:
        """build_embeddings should encode docs and persist the matrix to disk."""
        docs = [{"id": 1, "title": "A", "description": "desc"}]
        mock_embeddings = np.array([[1.0, 2.0, 3.0]])
        mock_model = MagicMock()
        mock_model.encode.return_value = mock_embeddings
        with (
            patch(
                "cli.core.semantic_search.SentenceTransformer",
                return_value=mock_model,
            ),
            patch("cli.core.semantic_search.np.save") as mock_save,
        ):
            ss = SemanticSearch()
            result = ss.build_embeddings(docs)  # type: ignore[arg-type]

        mock_model.encode.assert_called_once_with(["A: desc"], show_progress_bar=True)
        mock_save.assert_called_once()
        assert np.array_equal(result, mock_embeddings)

    def test_load_or_create_embeddings_returns_cached_when_file_exists(self) -> None:
        """load_or_create_embeddings returns cached embeddings when count matches."""
        docs = [{"id": 1, "title": "A", "description": "desc"}]
        mock_embeddings = MagicMock()
        mock_embeddings.__len__ = MagicMock(return_value=1)
        mock_model = MagicMock()
        mock_path = MagicMock()
        mock_path.is_file.return_value = True
        with (
            patch(
                "cli.core.semantic_search.SentenceTransformer",
                return_value=mock_model,
            ),
            patch.object(SemanticSearch, "EMBEDDINGS_FILE_PATH", mock_path),
            patch("cli.core.semantic_search.np.load", return_value=mock_embeddings),
        ):
            ss = SemanticSearch()
            result = ss.load_or_create_embeddings(docs)  # type: ignore[arg-type]

        assert result is mock_embeddings

    def test_load_or_create_embeddings_rebuilds_on_count_mismatch(self) -> None:
        """load_or_create_embeddings should rebuild when cached count doesn't match."""
        docs = [{"id": 1, "title": "A", "description": "desc"}]
        stale = np.zeros((99, 3))
        fresh = np.array([[1.0, 2.0, 3.0]])
        mock_model = MagicMock()
        mock_model.encode.return_value = fresh
        mock_path = MagicMock()
        mock_path.is_file.return_value = True
        with (
            patch(
                "cli.core.semantic_search.SentenceTransformer",
                return_value=mock_model,
            ),
            patch.object(SemanticSearch, "EMBEDDINGS_FILE_PATH", mock_path),
            patch("cli.core.semantic_search.np.load", return_value=stale),
            patch("cli.core.semantic_search.np.save"),
        ):
            ss = SemanticSearch()
            result = ss.load_or_create_embeddings(docs)  # type: ignore[arg-type]

        assert np.array_equal(result, fresh)

    def test_load_or_create_embeddings_builds_when_no_file(self) -> None:
        """load_or_create_embeddings should build embeddings when no file exists."""
        docs = [{"id": 1, "title": "A", "description": "desc"}]
        fresh = np.array([[1.0, 2.0, 3.0]])
        mock_model = MagicMock()
        mock_model.encode.return_value = fresh
        mock_path = MagicMock()
        mock_path.is_file.return_value = False
        with (
            patch(
                "cli.core.semantic_search.SentenceTransformer",
                return_value=mock_model,
            ),
            patch.object(SemanticSearch, "EMBEDDINGS_FILE_PATH", mock_path),
            patch("cli.core.semantic_search.np.save"),
        ):
            ss = SemanticSearch()
            result = ss.load_or_create_embeddings(docs)  # type: ignore[arg-type]

        assert np.array_equal(result, fresh)

    def test_search_raises_when_embeddings_not_loaded(self) -> None:
        """search should raise ValueError when embeddings have not been loaded."""
        mock_model = MagicMock()
        with patch(
            "cli.core.semantic_search.SentenceTransformer", return_value=mock_model
        ):
            ss = SemanticSearch()
            with pytest.raises(ValueError, match="No embeddings loaded"):
                ss.search("batman", 5)

    def test_search_returns_top_results_ranked_by_score(self) -> None:
        """search should return results sorted by descending cosine similarity."""
        docs = [
            {"id": 1, "title": "A", "description": "desc A"},
            {"id": 2, "title": "B", "description": "desc B"},
        ]
        mock_model = MagicMock()
        query_emb = np.array([1.0, 0.0])
        doc_emb_high = np.array([1.0, 0.0])
        doc_emb_low = np.array([0.0, 1.0])
        mock_model.encode.return_value = [query_emb]
        with patch(
            "cli.core.semantic_search.SentenceTransformer", return_value=mock_model
        ):
            ss = SemanticSearch()
            ss.documents = docs  # type: ignore[assignment]
            ss.embeddings = np.array(  # type: ignore[assignment]
                [doc_emb_high, doc_emb_low]
            )

            results = ss.search("query", 2)

        assert results[0]["title"] == "A"
        assert results[1]["title"] == "B"
        assert results[0]["score"] > results[1]["score"]

    def test_search_respects_limit(self) -> None:
        """search should return at most `limit` results."""
        docs = [
            {"id": 1, "title": "A", "description": "desc A"},
            {"id": 2, "title": "B", "description": "desc B"},
            {"id": 3, "title": "C", "description": "desc C"},
        ]
        mock_model = MagicMock()
        query_emb = np.array([1.0, 0.0])
        mock_model.encode.return_value = [query_emb]
        with patch(
            "cli.core.semantic_search.SentenceTransformer", return_value=mock_model
        ):
            ss = SemanticSearch()
            ss.documents = docs  # type: ignore[assignment]
            ss.embeddings = np.array(  # type: ignore[assignment]
                [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
            )

            results = ss.search("query", 2)

        assert len(results) == 2


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


class TestVerifyEmbeddings:
    """Tests for the verify_embeddings function."""

    def test_prints_document_count_and_shape(self, capsys: CaptureFixture[str]) -> None:
        """verify_embeddings should print number of docs and embedding shape."""
        mock_embeddings = MagicMock()
        mock_embeddings.shape = (2, 384)
        mock_docs = [
            {"id": 1, "title": "A", "description": "desc A"},
            {"id": 2, "title": "B", "description": "desc B"},
        ]
        mock_model = MagicMock()
        with (
            patch(
                "cli.core.semantic_search.SentenceTransformer",
                return_value=mock_model,
            ),
            patch("cli.core.semantic_search.get_movies", return_value=mock_docs),
            patch.object(
                SemanticSearch,
                "load_or_create_embeddings",
                return_value=mock_embeddings,
            ),
        ):
            verify_embeddings()

        out = capsys.readouterr().out
        assert "Number of docs:" in out
        assert "Embeddings shape:" in out


class TestEmbedQueryText:
    """Tests for the embed_query_text function."""

    def test_prints_query_and_embedding_info(self, capsys: CaptureFixture[str]) -> None:
        """embed_query_text should print the query, first 3 dimensions, and shape."""
        mock_embedding = MagicMock()
        mock_embedding.__getitem__ = MagicMock(return_value="[0.1, 0.2, 0.3]")
        mock_embedding.shape = (384,)
        mock_model = MagicMock()
        mock_model.encode.return_value = [mock_embedding]
        with patch(
            "cli.core.semantic_search.SentenceTransformer", return_value=mock_model
        ):
            embed_query_text("dark knight")

        out = capsys.readouterr().out
        assert "Query: dark knight" in out
        assert "First 3 dimensions:" in out
        assert "Shape:" in out


class TestChunkedSemanticSearch:
    """Tests for the ChunkedSemanticSearch class."""

    def test_initializes_chunk_state(self) -> None:
        """ChunkedSemanticSearch sets chunk_embeddings and chunk_metadata to None."""
        mock_model = MagicMock()
        with patch(
            "cli.core.semantic_search.SentenceTransformer", return_value=mock_model
        ):
            css = ChunkedSemanticSearch()
        assert css.chunk_embeddings is None
        assert css.chunk_metadata is None

    def test_returns_same_instance_on_multiple_calls(self) -> None:
        """Two ChunkedSemanticSearch() calls should return the identical object."""
        mock_model = MagicMock()
        with patch(
            "cli.core.semantic_search.SentenceTransformer", return_value=mock_model
        ):
            assert ChunkedSemanticSearch() is ChunkedSemanticSearch()

    def test_build_chunk_embeddings_skips_docs_without_description(self) -> None:
        """build_chunk_embeddings should skip docs that have an empty description."""
        docs = [
            {"id": 1, "title": "A", "description": ""},
            {"id": 2, "title": "B", "description": "A real sentence."},
        ]
        mock_embeddings = np.zeros((1, 3))
        mock_model = MagicMock()
        mock_model.encode.return_value = mock_embeddings
        with (
            patch(
                "cli.core.semantic_search.SentenceTransformer", return_value=mock_model
            ),
            patch("cli.core.semantic_search.np.save"),
            patch("builtins.open", mock_open()),
        ):
            css = ChunkedSemanticSearch()
            css.build_chunk_embeddings(docs)  # type: ignore[arg-type]

        encoded_chunks = mock_model.encode.call_args[0][0]
        assert all(chunk.strip() for chunk in encoded_chunks)

    def test_build_chunk_embeddings_encodes_and_saves(self) -> None:
        """build_chunk_embeddings should encode all chunks and persist to disk."""
        docs = [
            {"id": 1, "title": "A", "description": "First sentence. Second sentence."}
        ]
        mock_chunk_embeddings = np.zeros((2, 3))
        mock_model = MagicMock()
        mock_model.encode.return_value = mock_chunk_embeddings
        with (
            patch(
                "cli.core.semantic_search.SentenceTransformer", return_value=mock_model
            ),
            patch("cli.core.semantic_search.np.save"),
            patch("builtins.open", mock_open()),
        ):
            css = ChunkedSemanticSearch()
            result = css.build_chunk_embeddings(docs)  # type: ignore[arg-type]

        mock_model.encode.assert_called_once()
        assert np.array_equal(result, mock_chunk_embeddings)

    def test_load_or_create_chunk_embeddings_returns_cached(self) -> None:
        """load_or_create_chunk_embeddings returns cached when both files exist."""
        docs = [{"id": 1, "title": "A", "description": "desc"}]
        cached_embeddings = np.zeros((1, 3))
        mock_model = MagicMock()
        metadata_json = json.dumps(
            {"chunks": [{"doc_id": 1, "chunk_idx": 0, "total_chunks": 1}]}
        )
        mock_emb_path = MagicMock()
        mock_emb_path.is_file.return_value = True
        mock_meta_path = MagicMock()
        mock_meta_path.is_file.return_value = True
        with (
            patch(
                "cli.core.semantic_search.SentenceTransformer", return_value=mock_model
            ),
            patch.object(ChunkedSemanticSearch, "EMBEDDINGS_FILE_PATH", mock_emb_path),
            patch.object(
                ChunkedSemanticSearch, "CHUNK_METADATA_FILE_PATH", mock_meta_path
            ),
            patch("cli.core.semantic_search.np.load", return_value=cached_embeddings),
            patch("builtins.open", mock_open(read_data=metadata_json)),
        ):
            css = ChunkedSemanticSearch()
            result = css.load_or_create_chunk_embeddings(docs)  # type: ignore[arg-type]

        assert np.array_equal(result, cached_embeddings)

    def test_load_or_create_chunk_embeddings_builds_when_no_cache(self) -> None:
        """load_or_create_chunk_embeddings builds embeddings when no cache exists."""
        docs = [{"id": 1, "title": "A", "description": "First. Second."}]
        built_embeddings = np.zeros((2, 3))
        mock_model = MagicMock()
        mock_model.encode.return_value = built_embeddings
        mock_emb_path = MagicMock()
        mock_emb_path.is_file.return_value = False
        mock_meta_path = MagicMock()
        mock_meta_path.is_file.return_value = False
        with (
            patch(
                "cli.core.semantic_search.SentenceTransformer", return_value=mock_model
            ),
            patch.object(ChunkedSemanticSearch, "EMBEDDINGS_FILE_PATH", mock_emb_path),
            patch.object(
                ChunkedSemanticSearch, "CHUNK_METADATA_FILE_PATH", mock_meta_path
            ),
            patch("cli.core.semantic_search.np.save"),
            patch("builtins.open", mock_open()),
        ):
            css = ChunkedSemanticSearch()
            result = css.load_or_create_chunk_embeddings(docs)  # type: ignore[arg-type]

        assert np.array_equal(result, built_embeddings)


class TestEmbedChunks:
    """Tests for the embed_chunks function."""

    def test_prints_chunk_count(self, capsys: CaptureFixture[str]) -> None:
        """embed_chunks should print the number of chunk embeddings generated."""
        mock_embeddings = np.zeros((5, 3))
        mock_docs = [{"id": 1, "title": "A", "description": "desc"}]
        mock_model = MagicMock()
        with (
            patch(
                "cli.core.semantic_search.SentenceTransformer", return_value=mock_model
            ),
            patch("cli.core.semantic_search.get_movies", return_value=mock_docs),
            patch.object(
                ChunkedSemanticSearch,
                "load_or_create_chunk_embeddings",
                return_value=mock_embeddings,
            ),
        ):
            embed_chunks()

        out = capsys.readouterr().out
        assert "5" in out
        assert "chunked embeddings" in out
