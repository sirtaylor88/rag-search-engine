# Hybrid Search CLI

Entry point for the `hybrid_search_cli.py` CLI. Registers three subcommands —
`normalize`, `weighted-search`, and `rrf-search` — and dispatches parsed arguments
to the appropriate command class.

```{eval-rst}
.. automodule:: cli.hybrid_search_cli
   :members:
   :undoc-members:

.. seealso::

   :doc:`/api/hybrid_search` — core weighted and RRF search logic

   :doc:`/api/commands/search/hybrid_search_command` — ``WeightedSearchCommand`` and ``RRFSearchCommand``

   :doc:`/api/gemini_agent` — Gemini-powered query enhancement (``--enhance``) and re-ranking (``--rerank-method``)

   :doc:`/api/schemas` — ``WeightedSearchPayload``, ``RRFSearchPayload``, and request envelopes
```
