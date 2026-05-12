# Gemini Agent

Thin wrapper around the [Google Gemini API](https://ai.google.dev/) for
query-enhancement tasks. Two enhancement methods are available, each backed by
a `PromptPattern` template:

- **`spell`** — corrects only high-confidence typos, leaving the rest of the
  query unchanged.
- **`rewrite`** — expands a vague query into a concise, Google-style search
  phrase using common movie knowledge and genre conventions.

Passing `method=None` (the default) returns the original query immediately
without making any API call.

```{eval-rst}
.. note::

   ``GEMINI_API_KEY`` must be set in the environment (or in a ``.env`` file) before
   calling any function in this module.

.. automodule:: cli.api.gemini_agent
   :members:
   :undoc-members:
   :show-inheritance:

.. seealso::

   :doc:`/api/commands/search/hybrid_search_command` — ``RRFSearchCommand`` that calls ``enhance_query``

   :doc:`/api/constants` — ``GEMINI_MODEL`` and ``DEFAULT_ENHANCE_METHOD``
```
