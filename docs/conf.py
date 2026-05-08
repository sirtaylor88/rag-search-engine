"""Sphinx configuration for RAG Search Engine."""

# pylint: disable=invalid-name

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

project = "RAG Search Engine"
author = "Nhat Tai NGUYEN"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

html_theme = "furo"
html_title = "RAG Search Engine"

autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
