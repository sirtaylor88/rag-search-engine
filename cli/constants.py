"""Project-wide constants."""

from pathlib import Path

# * Cache
CACHE_DIR = "cache"
CACHE_DIR_PATH = Path(CACHE_DIR)

# * BM25
BM25_B = 0.75
BM25_K1 = 1.5

# * Search
SEARCH_LIMIT = 5

# * Chunk sizes
CHUNK_SIZE = 200
SEMANTIC_CHUNK_SIZE = 4

# * Regex
SENTENCE_SPLIT_PATTERN = r"(?<=[.!?])\s+"
