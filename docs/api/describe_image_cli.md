# Describe Image CLI

Entry point for `describe_image_cli.py`. Accepts two arguments:

- **`--image`** — path to an image file (JPEG, PNG, etc.).
- **`--query`** — the text query to enrich with image context.

The MIME type is detected via `mimetypes.guess_type`; when detection fails it
defaults to `"image/jpeg"`.

The pipeline:

1. Read the image file as raw bytes.
2. Call {func}`~cli.api.gemini_agent.describe_image` with the bytes, MIME
   type, and query.
3. Print `"Rewritten query: {text}"` (empty string when the model returns
   nothing).
4. If `usage_metadata` is present, print `"Total tokens: {total_token_count}"`.

```{eval-rst}
.. automodule:: cli.describe_image_cli
   :members:
   :undoc-members:

.. seealso::

   :doc:`/api/gemini_agent` — ``describe_image`` and ``DESCRIBE_IMAGE_PATTERN`` used for multimodal rewriting
```
