"""Single source of truth for the CrewAI/Ollama LLM instance.

The LLM object is created lazily and cached so we never rebuild it on every
request (avoids repeated connection setup and keeps latency low).
"""
from __future__ import annotations

import threading

from . import config

_llm = None
_lock = threading.Lock()


def get_llm():
    """Return a cached CrewAI LLM configured against the local Ollama server."""
    global _llm
    if _llm is not None:
        return _llm
    with _lock:
        if _llm is not None:
            return _llm
        from crewai import LLM

        _llm = LLM(
            model=f"ollama/{config.OLLAMA_MODEL}",
            base_url=config.OLLAMA_API_BASE,
            temperature=config.TEMPERATURE,
            max_tokens=512,
        )
    return _llm


def reset_llm() -> None:
    """Drop the cached LLM (used by tests / config reloads)."""
    global _llm
    with _lock:
        _llm = None