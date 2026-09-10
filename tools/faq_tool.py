"""CrewAI FAQ Retrieval Tool.

Wraps the threshold-based ChromaDB retriever. Returns only FAQ hits that pass
the configured similarity threshold; otherwise reports no relevant results so
the agent can fall back instead of hallucinating.
"""
from __future__ import annotations

import json

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from retrieval.retriever import retrieve_faq


class FAQRetrievalInput(BaseModel):
    question: str = Field(..., description="The customer's FAQ question.")


def retrieve(question: str) -> dict:
    result = retrieve_faq(question)
    return {
        "relevant": result.relevant,
        "top_similarity": round(result.top_similarity, 4),
        "hits": [
            {
                "faq_id": h.faq_id,
                "category": h.category,
                "question": h.question,
                "answer": h.answer,
                "similarity": round(h.similarity, 4),
                "source": h.source,
            }
            for h in result.hits
        ],
    }


class FAQRetrievalTool(BaseTool):
    name: str = "faq_retrieval"
    description: str = (
        "Retrieve relevant company FAQ answers for a customer question from the "
        "company policy knowledge base. Only returns results above the confidence "
        "threshold."
    )
    args_schema: type[BaseModel] = FAQRetrievalInput

    def _run(self, question: str) -> str:
        try:
            return json.dumps(retrieve(question), ensure_ascii=False)
        except Exception as exc:
            return json.dumps(
                {"relevant": False, "hits": [], "error": str(exc)},
                ensure_ascii=False,
            )