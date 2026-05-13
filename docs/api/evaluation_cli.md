# Evaluation CLI

Entry point for `evaluation_cli.py`. Loads the movie corpus and a golden
dataset from `data/golden_dataset.json`, runs RRF search for each test case,
computes Precision@k (`|retrieved ∩ relevant| / |retrieved|`), Recall@k
(`|retrieved ∩ relevant| / |relevant|`), and F1 (`2·P·R / (P+R)`), and
prints results sorted by F1 score (descending).

```{eval-rst}
.. automodule:: cli.evaluation_cli
   :members:
   :undoc-members:

.. seealso::

   :doc:`/api/hybrid_search` — ``HybridSearch.rrf_search`` used for retrieval

   :doc:`/api/constants` — ``DEFAULT_PRECISION_AT_K`` (default k)
```
