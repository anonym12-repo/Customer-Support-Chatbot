"""Threshold-based FAQ retrieval over ChromaDB.

Retrieval is never "return top-3 blindly": each result's similarity is computed
from its cosine distance and filtered against a configurable threshold. Below
threshold -> the retriever reports `relevant=False` so the pipeline can fall
back safely instead of hallucinating a policy.
"""
from __future__ import annotations

import logging
import time

from backend import config
from backend.schemas import FAQHit, RetrievalResult

logger = logging.getLogger(__name__)

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is not None:
        return _collection
    import chromadb

    from .embeddings import get_embedding_function

    _client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
    _collection = _client.get_collection(
        name=config.COLLECTION_NAME,
        embedding_function=get_embedding_function(),
    )
    return _collection


def reset_collection_cache() -> None:
    global _client, _collection
    _client = None
    _collection = None


def _distance_to_similarity(distance: float) -> float:
    """Convert a ChromaDB cosine distance to a cosine similarity.

    ChromaDB (hnsw:space=cosine) returns distance = 1 - cosine_similarity,
    in the range 0..2. So similarity = 1 - distance, in the range -1..1;
    we clamp to [0, 1] because negative similarity is never relevant anyway.
    """
    sim = 1.0 - distance
    return max(0.0, min(1.0, sim))


def index_available() -> bool:
    """True if the FAQ collection exists and can be opened (no embedding call)."""
    try:
        _get_collection()
        return True
    except Exception:
        return False


def retrieve_faq(question: str, k: int | None = None) -> RetrievalResult:
    """Retrieve relevant FAQ hits for a question, applying a similarity threshold."""
    start = time.perf_counter()
    k = k or config.FAQ_RETRIEVAL_K
    try:
        collection = _get_collection()
    except Exception as exc:
        logger.error("Chroma collection unavailable: %s", exc)
        return RetrievalResult(
            relevant=False, hits=[], latency_ms=(time.perf_counter() - start) * 1000
        )

    try:
        results = collection.query(query_texts=[question], n_results=k)
    except Exception as exc:
        logger.error("FAQ query failed: %s", exc)
        return RetrievalResult(
            relevant=False, hits=[], latency_ms=(time.perf_counter() - start) * 1000
        )

    hits: list[FAQHit] = []
    distances = (results.get("distances") or [[]])[0]
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]

    for dist, doc, meta in zip(distances, documents, metadatas):
        meta = meta or {}
        sim = _distance_to_similarity(float(dist))
        hits.append(
            FAQHit(
                faq_id=meta.get("faq_id", "unknown"),
                question=meta.get("question", ""),
                answer=meta.get("answer") or doc or "",
                category=meta.get("category", "uncategorized"),
                source=meta.get("source", "Company_FAQ.pdf"),
                similarity=sim,
            )
        )

    relevant_hits = [h for h in hits if h.similarity >= config.FAQ_SIMILARITY_THRESHOLD]
    top_sim = hits[0].similarity if hits else 0.0
    latency_ms = (time.perf_counter() - start) * 1000

    return RetrievalResult(
        relevant=len(relevant_hits) > 0,
        hits=relevant_hits,
        top_similarity=top_sim,
        latency_ms=latency_ms,
    )