# Multimodal Search CLI

Entry point for `multimodal_search_cli.py`. Registers one subcommand:

- **`verify_image_embedding`** via
  {class}`~cli.commands.verify_commands.VerifyImageEmbeddingCommand` —
  embeds an image using the CLIP model and prints the embedding size.

The subcommand accepts one positional argument:

- **`image_path`** — path to the image file to embed.

```{eval-rst}
.. automodule:: cli.multimodal_search_cli
   :members:
   :undoc-members:

.. seealso::

   :doc:`/api/multimodal_search` — ``MultimodalSearch`` and ``verify_image_embedding``

   :doc:`/api/commands/verify` — ``VerifyImageEmbeddingCommand`` implementation
```
