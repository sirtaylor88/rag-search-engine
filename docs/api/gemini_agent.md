# Gemini Agent

Thin wrapper around the [Google Gemini API](https://ai.google.dev/) for
query-enhancement tasks. Currently supports spell-correction (`method="spell"`),
which sends the raw query to Gemini with a strict prompt that corrects only
high-confidence typos.

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
