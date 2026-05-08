"""Tests for cli.keyword_search_cli."""

# pylint: disable=redefined-outer-name

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pytest import CaptureFixture

from cli.commands.build_command import get_movies
from cli.keyword_search_cli import main


def test_search_command_outputs_results(capsys: CaptureFixture[str]) -> None:
    """The search command should print a header and matching results."""
    mock_idx = MagicMock()
    mock_idx.search.return_value = [1]
    mock_idx.docmap = {1: {"title": "Batman"}}
    with (
        patch("sys.argv", ["cli", "search", "batman"]),
        patch("cli.commands.base.InvertedIndex", return_value=mock_idx),
    ):
        main()
    out = capsys.readouterr().out
    assert "Searching for: batman" in out
    assert "Batman" in out


def test_search_missing_cache_prints_error_and_exits(
    capsys: CaptureFixture[str],
) -> None:
    """A missing cache file should print an error message and exit with code 1."""
    with (
        patch("cli.core.keyword_search.InvertedIndex.load", side_effect=OSError),
        patch("sys.argv", ["cli", "search", "batman"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 1
    assert (
        capsys.readouterr().out
        == "Cannot load cache data. Please run build command first.\n"
    )


def test_bm25search_command_outputs_results(capsys: CaptureFixture[str]) -> None:
    """The bm25search command should print a header and ranked results."""
    mock_idx = MagicMock()
    mock_idx.bm25_search.return_value = [(1, 2.5)]
    mock_idx.docmap = {1: {"title": "Batman"}}
    with (
        patch("sys.argv", ["cli", "bm25search", "batman"]),
        patch("cli.commands.base.InvertedIndex", return_value=mock_idx),
    ):
        main()
    out = capsys.readouterr().out
    assert "Searching for: batman" in out
    assert "Batman" in out
    assert "2.50" in out


def test_bm25search_missing_cache_prints_error_and_exits(
    capsys: CaptureFixture[str],
) -> None:
    """A missing cache file should print an error message and exit with code 1."""
    with (
        patch("cli.core.keyword_search.InvertedIndex.load", side_effect=OSError),
        patch("sys.argv", ["cli", "bm25search", "batman"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 1
    assert (
        capsys.readouterr().out
        == "Cannot load cache data. Please run build command first.\n"
    )


def test_build_command_builds_and_saves_index(capsys: CaptureFixture[str]) -> None:
    """The build command should build and save the index and print status messages."""
    movies = [{"id": 1, "title": "Test", "description": ""}]
    with (
        patch("sys.argv", ["cli", "build"]),
        patch("cli.commands.build_command.get_movies", return_value=movies),
        patch("cli.core.keyword_search.InvertedIndex.build"),
        patch("cli.core.keyword_search.InvertedIndex.save"),
    ):
        main()
    out = capsys.readouterr().out
    assert "Building inverted index" in out
    assert "has been built" in out


def test_no_command_prints_help(capsys: CaptureFixture[str]) -> None:
    """Running without a subcommand should print the help message."""
    with patch("sys.argv", ["cli"]):
        main()
    assert capsys.readouterr().out != ""


def test_get_movies_loads_file(tmp_path: Path) -> None:
    """get_movies should parse a JSON file and return the 'movies' list."""
    data = {"movies": [{"id": 1, "title": "Test", "description": ""}]}
    path = tmp_path / "movies.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert get_movies(str(path)) == data["movies"]


def test_tf_command_outputs_result(capsys: CaptureFixture[str]) -> None:
    """The tf command should load the index and print the term frequency."""
    with (
        patch("sys.argv", ["cli", "tf", "1", "batman"]),
        patch("cli.core.keyword_search.InvertedIndex.load"),
        patch("cli.core.keyword_search.InvertedIndex.get_tf", return_value=3),
    ):
        main()
    assert "batman" in capsys.readouterr().out


def test_tf_missing_cache_prints_error_and_exits(
    capsys: CaptureFixture[str],
) -> None:
    """A missing cache file should print an error message and exit with code 1."""
    with (
        patch("cli.core.keyword_search.InvertedIndex.load", side_effect=OSError),
        patch("sys.argv", ["cli", "tf", "1", "batman"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 1
    assert (
        capsys.readouterr().out
        == "Cannot load cache data. Please run build command first.\n"
    )


def test_idf_command_outputs_result(capsys: CaptureFixture[str]) -> None:
    """The idf command should load the index and print the IDF value."""
    with (
        patch("sys.argv", ["cli", "idf", "batman"]),
        patch("cli.core.keyword_search.InvertedIndex.load"),
    ):
        main()
    assert "batman" in capsys.readouterr().out


def test_idf_missing_cache_prints_error_and_exits(
    capsys: CaptureFixture[str],
) -> None:
    """A missing cache file should print an error message and exit with code 1."""
    with (
        patch("cli.core.keyword_search.InvertedIndex.load", side_effect=OSError),
        patch("sys.argv", ["cli", "idf", "batman"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 1
    assert (
        capsys.readouterr().out
        == "Cannot load cache data. Please run build command first.\n"
    )


def test_tfidf_command_outputs_result(capsys: CaptureFixture[str]) -> None:
    """The tfidf command should load the index and print the TF-IDF score."""
    with (
        patch("sys.argv", ["cli", "tfidf", "1", "batman"]),
        patch("cli.core.keyword_search.InvertedIndex.load"),
        patch("cli.core.keyword_search.InvertedIndex.get_idf", return_value=1.5),
        patch("cli.core.keyword_search.InvertedIndex.get_tf", return_value=2),
    ):
        main()
    assert "batman" in capsys.readouterr().out


def test_tfidf_missing_cache_prints_error_and_exits(
    capsys: CaptureFixture[str],
) -> None:
    """A missing cache file should print an error message and exit with code 1."""
    with (
        patch("cli.core.keyword_search.InvertedIndex.load", side_effect=OSError),
        patch("sys.argv", ["cli", "tfidf", "1", "batman"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 1
    assert (
        capsys.readouterr().out
        == "Cannot load cache data. Please run build command first.\n"
    )


def test_bm25idf_command_outputs_result(capsys: CaptureFixture[str]) -> None:
    """The bm25idf command should load the index and print the BM25 IDF value."""
    with (
        patch("sys.argv", ["cli", "bm25idf", "batman"]),
        patch("cli.core.keyword_search.InvertedIndex.load"),
    ):
        main()
    assert "batman" in capsys.readouterr().out


def test_bm25idf_missing_cache_prints_error_and_exits(
    capsys: CaptureFixture[str],
) -> None:
    """A missing cache file should print an error message and exit with code 1."""
    with (
        patch("cli.core.keyword_search.InvertedIndex.load", side_effect=OSError),
        patch("sys.argv", ["cli", "bm25idf", "batman"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 1
    assert (
        capsys.readouterr().out
        == "Cannot load cache data. Please run build command first.\n"
    )


def test_bm25tf_command_outputs_result(capsys: CaptureFixture[str]) -> None:
    """The bm25tf command should load the index and print the BM25 TF score."""
    with (
        patch("sys.argv", ["cli", "bm25tf", "1", "batman"]),
        patch("cli.core.keyword_search.InvertedIndex.load"),
        patch("cli.core.keyword_search.InvertedIndex.get_bm25_tf", return_value=1.6),
    ):
        main()
    assert "batman" in capsys.readouterr().out


def test_bm25tf_missing_cache_prints_error_and_exits(
    capsys: CaptureFixture[str],
) -> None:
    """A missing cache file should print an error message and exit with code 1."""
    with (
        patch("cli.core.keyword_search.InvertedIndex.load", side_effect=OSError),
        patch("sys.argv", ["cli", "bm25tf", "1", "batman"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 1
    assert (
        capsys.readouterr().out
        == "Cannot load cache data. Please run build command first.\n"
    )


def test_build_missing_data_prints_error_and_exits(
    capsys: CaptureFixture[str],
) -> None:
    """A missing data file should print an error message and exit with code 1."""
    with (
        patch("cli.commands.build_command.get_movies", side_effect=OSError),
        patch("sys.argv", ["cli", "build", "--data-path", "missing.json"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 1
    assert capsys.readouterr().out == "Cannot build movies data.\n"
