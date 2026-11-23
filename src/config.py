import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Base directories (Pathlib handles Windows/Linux paths cleanly)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "data_index"
CHROMA_DIR = INDEX_DIR / "chroma"
MANIFEST_PATH = INDEX_DIR / "manifest.json"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
CACHE_DB_PATH = INDEX_DIR / "cache.db"
FLASHRANK_CACHE_DIR = BASE_DIR / "flashrank_cache"

# Ensure required directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Load .env from project root
load_dotenv(BASE_DIR / ".env")

# Vector DB & Chunking parameters
COLLECTION_NAME = "rag_documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_CHUNK_CHARS = 40

# Reranker & LLM configuration (fastest low-latency models first)
RERANKER_MODEL = "ms-marco-TinyBERT-L-2-v2"
MODEL_FALLBACK_CHAIN = [
    ("gemini-3.5-flash-lite", "minimal"),
    ("gemini-3.1-flash-lite", "minimal"),
    ("gemini-3.6-flash", "minimal"),
    ("gemini-3.8-flash", "low"),
]


def get_api_key() -> str:
    """Return GEMINI_API_KEY or raise a clear configuration error."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError(
            "GEMINI_API_KEY is missing or still set to the placeholder. "
            "Add your real key to the .env file."
        )
    return api_key


def get_logger(name: str) -> logging.Logger:
    """Return a clean, timestamped logger for production modules."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)