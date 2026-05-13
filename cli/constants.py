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
CHUNKED_SEARCH_LIMIT = 10
DEFAULT_ALPHA = 0.5
DEFAULT_K = 60

# * Chunk sizes
CHUNK_SIZE = 200
SEMANTIC_CHUNK_SIZE = 4

# * Regex
SENTENCE_SPLIT_PATTERN = r"(?<=[.!?])\s+"

# * Embedding model
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# * Cross encoder model
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-TinyBERT-L2-v2"

# * Score
SCORE_PRECISION = 4

# * AI Agent
GEMINI_MODEL = "gemma-4-31b-it"

# * Precision
DEFAULT_PRECISION_AT_K = 5
