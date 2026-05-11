# Hybrid Search Commands

Command classes for the `weighted-search` and `rrf-search` subcommands. Both
inherit from `BaseHybridSearchCommand[P]`, which owns the shared
load → search → print loop. Subclasses implement only `_search` (calls the
appropriate {class}`~cli.core.hybrid_search.HybridSearch` method) and
`_format_scores` (formats the per-result score line).

```{eval-rst}
.. automodule:: cli.commands.search.hybrid_search_commands
   :members:
   :undoc-members:
   :show-inheritance:

.. seealso::

   :doc:`/api/hybrid_search` — ``HybridSearch`` singleton called by both commands

   :doc:`/api/gemini_agent` — ``enhance_query`` used by ``RRFSearchCommand``

   :doc:`/api/schemas` — ``WeightedSearchPayload`` and ``RRFSearchPayload``
```
