# Keyword Search CLI

Entry point for `keyword_search_cli.py`. Registers eight subcommands for
building and querying the inverted index:

| Subcommand | Command class | Description |
|---|---|---|
| `build` | `BuildCommand` | Build and persist the inverted index from the movie dataset |
| `search` | `SearchCommand` | Unranked token-overlap search |
| `bm25search` | `BM25SearchCommand` | BM25-ranked keyword search |
| `tf` | `ComputeTFCommand` | Raw term frequency for a term in a document |
| `idf` | `ComputeIDFCommand` | Smoothed inverse document frequency for a term |
| `tfidf` | `ComputeTFIDFCommand` | TF-IDF score for a term in a document |
| `bm25idf` | `ComputeBM25IDFCommand` | BM25 IDF score for a term |
| `bm25tf` | `ComputeBM25TFCommand` | Length-normalised BM25 TF score |

```{eval-rst}
.. automodule:: cli.keyword_search_cli
   :members:
   :undoc-members:

.. seealso::

   :doc:`/api/inverted_index` — ``InvertedIndex`` singleton used by all subcommands

   :doc:`/api/commands/build` — ``BuildCommand`` implementation

   :doc:`/api/commands/search/keyword_search_commands` — ``SearchCommand`` and ``BM25SearchCommand``

   :doc:`/api/commands/compute/tf` — ``ComputeTFCommand``

   :doc:`/api/commands/compute/idf` — ``ComputeIDFCommand``

   :doc:`/api/commands/compute/tfidf` — ``ComputeTFIDFCommand``

   :doc:`/api/commands/compute/bm25_idf` — ``ComputeBM25IDFCommand``

   :doc:`/api/commands/compute/bm25_tf` — ``ComputeBM25TFCommand``
```
