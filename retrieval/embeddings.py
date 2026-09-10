"""Ollama embedding function for ChromaDB.

Uses ChromaDB's built-in OllamaEmbeddingFunction backed by nomic-embed-text.
The function is cached so we never rebuild it per request.
"""
from __future__ import annotations

import threading

from backend import config

_ef = None
_lock = threading.Lock()


def get_embedding_function():
    """Return a cached ChromaDB embedding function backed by Ollama."""
    global _ef
    if _ef is not None:
        return _ef
    with _lock:
        if _ef is not None:
            return _ef
        from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

        _ef = OllamaEmbeddingFunction(
            url=config.OLLAMA_API_BASE,
            model_name=config.EMBED_MODEL,
        )
    return _ef