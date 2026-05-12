# Schemas

## Payloads

Top-level type aliases used across payload models and CLI argument parsers:

- **`EnhanceMethod`** — `Literal["spell", "rewrite", "expand"]`. Selects the
  query-enhancement prompt in {func}`~cli.api.gemini_agent.enhance_query`.
- **`ReRankeMethod`** — `Literal["individual", "batch"]`. Selects the
  re-ranking prompt in {func}`~cli.api.gemini_agent.rerank_query`.
  - `individual` — one API call per candidate, returns a 0–10 score string.
  - `batch` — single API call for all candidates, returns a JSON-ordered list
    of document IDs.
- **`PositiveFloat`** — `Annotated[float, Field(gt=0)]`. Used by
  `ScoreListPayload`.

```{eval-rst}
.. automodule:: cli.schemas.payloads
   :members:
   :undoc-members:
   :show-inheritance:

.. seealso::

   :doc:`/api/gemini_agent` — ``enhance_query`` and ``rerank_query`` consume ``EnhanceMethod`` and ``ReRankeMethod``

   :doc:`/api/commands/search/hybrid_search_command` — ``RRFSearchCommand`` that uses ``RRFSearchPayload``
```

## Requests

```{eval-rst}
.. automodule:: cli.schemas.requests
   :members:
   :undoc-members:
   :show-inheritance:

.. seealso::

   :doc:`/api/hybrid_search_cli` — dispatches requests to hybrid search commands

   :doc:`/api/constants` — default values used by payload fields
```
