"""Tests for cli.keyword_search_cli."""

# pylint: disable=redefined-outer-name

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from pytest import CaptureFixture

from cli.commands.build_command import get_movies
from cli.commands.search_command import display_best_results
from cli.inverted_index import Document, InvertedIndex
from cli.keyword_search_cli import main


@pytest.fixture()
def inverted_index() -> InvertedIndex:
    """InvertedIndex built from a small movie list for search tests."""
    movies = [
        Document(id=i, title=title, description="")
        for i, title in enumerate(
            [
                "Batman Begins",
                "Batman Forever",
                "Batman Returns",
                "Batman & Robin",
                "Batman v Superman",
                "The Dark Knight",
                "Inception",
            ],
            start=1,
        )
    ]
    idx = InvertedIndex()
    idx.build(movies)
    return idx


@pytest.mark.parametrize("query", ["batman", "BATMAN"])
def test_returns_matching_results(
    inverted_index: InvertedIndex, capsys: CaptureFixture[str], query: str
) -> None:
    """Only matching titles should appear, capped at five, case-insensitively."""
    display_best_results(query, inverted_index)
    output = capsys.readouterr().out.splitlines()
    assert len(output) == 5
    assert all("batman" in line.lower() for line in output)


def test_no_matches_prints_nothing(
    inverted_index: InvertedIndex, capsys: CaptureFixture[str]
) -> None:
    """No output should be produced when the query matches no titles."""
    display_best_results("nomatch", inverted_index)
    output = capsys.readouterr().out
    assert output == ""


def test_fewer_than_nb_of_results_matches(
    inverted_index: InvertedIndex, capsys: CaptureFixture[str]
) -> None:
    """All matches should be shown when there are fewer than nb_of_results."""
    display_best_results("inception", inverted_index)
    output = capsys.readouterr().out.splitlines()
    assert len(output) == 1
    assert "Inception" in output[0]


def test_result_lines_are_numbered(
    inverted_index: InvertedIndex, capsys: CaptureFixture[str]
) -> None:
    """Each result line should be prefixed with the movie's doc ID."""
    display_best_results("batman", inverted_index)
    output = capsys.readouterr().out.splitlines()
    for i, line in enumerate(output, start=1):
        assert line.startswith(f"{i}.")


def test_nb_of_results_respected(
    inverted_index: InvertedIndex, capsys: CaptureFixture[str]
) -> None:
    """nb_of_results should cap the number of results displayed."""
    display_best_results("batman", inverted_index, nb_of_results=3)
    assert len(capsys.readouterr().out.splitlines()) == 3


def test_search_missing_cache_prints_error_and_exits(
    capsys: CaptureFixture[str],
) -> None:
    """A missing cache file should print an error message and exit with code 1."""
    with (
        patch("cli.commands.search_command.InvertedIndex.load", side_effect=OSError),
        patch("sys.argv", ["cli", "search", "batman"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 1
    assert (
        capsys.readouterr().out
        == "Cannot load cache data. Please run build command first.\n"
    )


def test_early_break_when_docs_limit_reached(capsys: CaptureFixture[str]) -> None:
    """The query loop should stop early once the result limit is reached."""
    movies = [
        Document(id=i, title="Batman", description="Superman") for i in range(1, 7)
    ]
    idx = InvertedIndex()
    idx.build(movies)
    display_best_results("batman superman", idx)
    assert len(capsys.readouterr().out.splitlines()) == 5


def test_search_command_outputs_results(capsys: CaptureFixture[str]) -> None:
    """The search command should print a header and call display_best_results."""
    with (
        patch("sys.argv", ["cli", "search", "batman"]),
        patch("cli.commands.search_command.InvertedIndex.load"),
        patch("cli.commands.search_command.display_best_results") as mock_display,
    ):
        main()
    assert "Searching for: batman" in capsys.readouterr().out
    mock_display.assert_called_once()


def test_build_command_builds_and_saves_index(capsys: CaptureFixture[str]) -> None:
    """The build command should build and save the index and print status messages."""
    movies = [{"id": 1, "title": "Test", "description": ""}]
    with (
        patch("sys.argv", ["cli", "build"]),
        patch("cli.commands.build_command.get_movies", return_value=movies),
        patch("cli.inverted_index.InvertedIndex.build"),
        patch("cli.inverted_index.InvertedIndex.save"),
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
        patch("cli.inverted_index.InvertedIndex.load"),
        patch("cli.inverted_index.InvertedIndex.get_tf", return_value=3),
    ):
        main()
    assert "batman" in capsys.readouterr().out


def test_tf_missing_cache_prints_error_and_exits(
    capsys: CaptureFixture[str],
) -> None:
    """A missing cache file should print an error message and exit with code 1."""
    with (
        patch("cli.inverted_index.InvertedIndex.load", side_effect=OSError),
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
        patch("cli.inverted_index.InvertedIndex.load"),
    ):
        main()
    assert "batman" in capsys.readouterr().out


def test_idf_missing_cache_prints_error_and_exits(
    capsys: CaptureFixture[str],
) -> None:
    """A missing cache file should print an error message and exit with code 1."""
    with (
        patch("cli.inverted_index.InvertedIndex.load", side_effect=OSError),
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
        patch("cli.inverted_index.InvertedIndex.load"),
        patch("cli.inverted_index.InvertedIndex.get_idf", return_value=1.5),
        patch("cli.inverted_index.InvertedIndex.get_tf", return_value=2),
    ):
        main()
    assert "batman" in capsys.readouterr().out


def test_tfidf_missing_cache_prints_error_and_exits(
    capsys: CaptureFixture[str],
) -> None:
    """A missing cache file should print an error message and exit with code 1."""
    with (
        patch("cli.inverted_index.InvertedIndex.load", side_effect=OSError),
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
        patch("cli.inverted_index.InvertedIndex.load"),
    ):
        main()
    assert "batman" in capsys.readouterr().out


def test_bm25idf_missing_cache_prints_error_and_exits(
    capsys: CaptureFixture[str],
) -> None:
    """A missing cache file should print an error message and exit with code 1."""
    with (
        patch("cli.inverted_index.InvertedIndex.load", side_effect=OSError),
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
        patch("cli.inverted_index.InvertedIndex.load"),
        patch("cli.inverted_index.InvertedIndex.get_tf", return_value=2),
    ):
        main()
    assert "batman" in capsys.readouterr().out


def test_bm25tf_missing_cache_prints_error_and_exits(
    capsys: CaptureFixture[str],
) -> None:
    """A missing cache file should print an error message and exit with code 1."""
    with (
        patch("cli.inverted_index.InvertedIndex.load", side_effect=OSError),
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
