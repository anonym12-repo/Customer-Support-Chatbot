"""Central configuration for the Customer Support AI system.

All tunable values are read from environment variables (loaded via python-dotenv)
with sensible defaults so the application runs locally out of the box.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present (never committed).
load_dotenv()

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
PROMPTS_DIR = Path(os.getenv("PROMPTS_DIR", str(BASE_DIR / "prompts")))
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", str(BASE_DIR / "chroma_db")))
FAQ_PDF_PATH = DATA_DIR / "Company_FAQ.pdf"
ORDERS_CSV = DATA_DIR / "sample_orders.csv"
PRODUCTS_CSV = DATA_DIR / "sample_products.csv"
ESCALATIONS_FILE = DATA_DIR / "escalations.json"

# --- Ollama / LLM ---
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "90"))

# --- Embeddings / Retrieval ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "company_faq")
FAQ_RETRIEVAL_K = int(os.getenv("FAQ_RETRIEVAL_K", "3"))
# Minimum cosine similarity (0-1) for a FAQ result to be considered relevant.
# ChromaDB (cosine space) distance = 1 - cosine similarity, and the retriever
# converts back: similarity = 1 - distance. So 0.55 means cosine similarity
# >= 0.55 (distance <= 0.45). Raise this to be stricter, lower to be more
# permissive.
FAQ_SIMILARITY_THRESHOLD = float(os.getenv("FAQ_SIMILARITY_THRESHOLD", "0.55"))

# --- Misc ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def as_dict() -> dict:
    """Return the non-secret configuration as a dictionary (for debugging/README)."""
    return {
        "OLLAMA_MODEL": OLLAMA_MODEL,
        "OLLAMA_API_BASE": OLLAMA_API_BASE,
        "EMBED_MODEL": EMBED_MODEL,
        "COLLECTION_NAME": COLLECTION_NAME,
        "FAQ_RETRIEVAL_K": FAQ_RETRIEVAL_K,
        "FAQ_SIMILARITY_THRESHOLD": FAQ_SIMILARITY_THRESHOLD,
        "TEMPERATURE": TEMPERATURE,
        "CHROMA_PATH": str(CHROMA_PATH),
    }