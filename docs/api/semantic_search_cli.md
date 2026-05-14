# Semantic Search CLI

Entry point for `semantic_search_cli.py`. Registers nine subcommands:

| Subcommand | Command class | Description |
|---|---|---|
| `verify` | `VerifyCommand` | Load the sentence-transformer model and print its properties |
| `verify_embeddings` | `VerifyEmbeddingsCommand` | Load or build corpus embeddings and print their shape |
| `embed_text` | `EmbedTextCommand` | Encode a text string and print its embedding info |
| `embed_query` | `EmbedQueryCommand` | Encode a query string and print its full embedding shape |
| `embed_chunks` | `EmbedChunksCommand` | Load or create sentence-level chunk embeddings for the corpus |
| `search` | `SemanticSearchCommand` | Rank movies by cosine similarity to the query |
| `search_chunked` | `SearchChunkedCommand` | Rank movies by max chunk cosine similarity |
| `chunk` | `ChunkCommand` | Split text into fixed-size word groups |
| `semantic_chunk` | `SemanticChunkCommand` | Split text into fixed-size sentence groups |

```{eval-rst}
.. automodule:: cli.semantic_search_cli
   :members:
   :undoc-members:

.. seealso::

   :doc:`/api/semantic_search` — ``SemanticSearch`` and ``ChunkedSemanticSearch`` backends

   :doc:`/api/commands/verify` — ``VerifyCommand`` and ``VerifyEmbeddingsCommand``

   :doc:`/api/commands/embed` — ``EmbedTextCommand``, ``EmbedQueryCommand``, ``EmbedChunksCommand``

   :doc:`/api/commands/search/semantic_search_command` — ``SemanticSearchCommand`` and ``SearchChunkedCommand``

   :doc:`/api/commands/compute/chunk` — ``ChunkCommand`` and ``SemanticChunkCommand``
```
