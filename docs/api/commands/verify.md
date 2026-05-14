# Verify Command

`BaseVerifyCommand` is an abstract base whose `run()` delegates to `_verify()`.
Its `add_arguments` is a no-op — no arguments are required for plain verify commands.

- **`VerifyCommand`** — calls `verify_model()` to load the semantic search model
  and print its properties.
- **`VerifyEmbeddingsCommand`** — calls `verify_embeddings()` to load or build
  corpus embeddings and print their shape.

For image-related verification see
{doc}`/api/commands/search/multimodal_search_commands`.

```{eval-rst}
.. automodule:: cli.commands.verify_commands
   :members:
   :undoc-members:
   :show-inheritance:

.. seealso::

   :doc:`/api/commands/search/multimodal_search_commands` — ``VerifyImageEmbeddingCommand`` and ``ImageSearchCommand``

   :doc:`/api/semantic_search` — ``verify_model`` and ``verify_embeddings``
```
