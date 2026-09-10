"""FAQ ingestion pipeline.

Reads Company_FAQ.pdf with PyPDF, extracts structured FAQ question/answer units,
embeds them with Ollama's nomic-embed-text and stores them in a persistent
ChromaDB collection with metadata (faq_id, question, category, source).

Run once before starting the app:
    python -m retrieval.index_faq
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from backend import config

logger = logging.getLogger(__name__)

# A FAQ unit looks like:
#   Q: What is your return policy?
#   A: You can return items within 30 days...
QA_RE = re.compile(r"Q[:.]\s*(.+?)\s*\n+\s*A[:.]\s*(.+?)(?=\n\s*\n|\nQ[:.]|\Z)", re.S)


def _extract_faqs_from_pdf(pdf_path: Path) -> list[dict]:
    """Extract FAQ units from the PDF, preserving question/answer pairs."""
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover
        from PyPDF2 import PdfReader  # type: ignore

    if not pdf_path.exists():
        raise FileNotFoundError(f"FAQ PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)

    faqs: list[dict] = []
    for idx, match in enumerate(QA_RE.finditer(text), start=1):
        question = match.group(1).strip().replace("\n", " ")
        answer = match.group(2).strip().replace("\n", " ")
        faqs.append({
            "faq_id": f"faq_{idx:03d}",
            "question": question,
            "answer": answer,
            "category": _categorize(question),
            "source": pdf_path.name,
        })
    return faqs


def _categorize(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ("ship", "delivery", "deliver", "international", "po box")):
        return "shipping"
    if any(w in q for w in ("return", "refund", "exchange")):
        return "returns"
    if any(w in q for w in ("payment", "pay", "method")):
        return "payments"
    if any(w in q for w in ("track", "order status", "status", "cancel",
                            "damaged", "replacement", "warranty")):
        return "orders"
    return "general"


def build_index(pdf_path: Path | None = None) -> int:
    """Build (or rebuild) the ChromaDB FAQ collection. Returns the FAQ count."""
    import chromadb

    from .embeddings import get_embedding_function

    pdf_path = pdf_path or config.FAQ_PDF_PATH
    faqs = _extract_faqs_from_pdf(pdf_path)
    if not faqs:
        raise RuntimeError("No FAQ entries extracted from PDF. Check the PDF format.")

    client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
    # Recreate the collection for a clean index.
    try:
        client.delete_collection(name=config.COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=get_embedding_function(),
    )

    # Embed "question + answer" as the document: user queries paraphrase the
    # FAQ questions, so matching against questions (not answers alone)
    # materially improves retrieval quality.
    collection.add(
        ids=[f["faq_id"] for f in faqs],
        documents=[f"Q: {f['question']}\nA: {f['answer']}" for f in faqs],
        metadatas=[
            {
                "faq_id": f["faq_id"],
                "question": f["question"],
                "answer": f["answer"],
                "category": f["category"],
                "source": f["source"],
            }
            for f in faqs
        ],
    )
    logger.info("Indexed %d FAQ entries into %s", len(faqs), config.COLLECTION_NAME)
    return len(faqs)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    count = build_index()
    print(f"Indexed {count} FAQ entries.")