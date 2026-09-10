"""Lightweight response guard.

Deterministic validation of the final answer *before* it reaches the user.
No extra LLM call is used for validation -- only cheap string/rule checks.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Phrases that would leak internal implementation details to the customer.
LEAK_PATTERNS = [
    re.compile(r"\b(system prompt|chain of thought|chain-of-thought)\b", re.I),
    re.compile(r"\bI (am|was) (instructed|told|programmed)\b", re.I),
    re.compile(r"\b(tool|agent|crew|task) (called|execution|output)\b", re.I),
    re.compile(r"\bFAQ_RETRIEVAL|ChromaDB|embedding|vector database\b", re.I),
]

SAFE_FALLBACK = (
    "I'm sorry, I had trouble preparing an answer. "
    "Please try rephrasing your question or contact customer support."
)

OUT_OF_SCOPE_FALLBACK = (
    "I'm designed to help with this company's orders and policies. "
    "I can't help with that request."
)

NOT_FOUND_FALLBACK = (
    "I couldn't find that information in the company's available policies. "
    "Please contact customer support for further assistance."
)


def order_not_found_message(order_number: int) -> str:
    """Deterministic, reusable decline for a non-existent order number."""
    return (
        f"I couldn't find an order with number {order_number} in our system. "
        "Please double-check the order number and try again."
    )


def _is_empty(answer: str) -> bool:
    return not answer or not answer.strip()


def _leaks_internals(answer: str) -> bool:
    return any(p.search(answer) for p in LEAK_PATTERNS)


def _claims_unfound_order(answer: str, order_number: int | None,
                           order_found: bool | None) -> bool:
    """If the order was NOT found but the answer invents status/details for it."""
    if order_found is None or order_found is True or order_number is None:
        return False
    # The answer mentions the order number alongside status-like words.
    status_words = re.compile(
        r"\b(shipped|pending|processing|cancelled|delivered|on its way|"
        r"arrive|tracking|total of|\$)\b", re.I
    )
    mentions_order = str(order_number) in answer
    return mentions_order and bool(status_words.search(answer))


def validate(answer: str, *, intent: str, order_number: int | None = None,
              order_found: bool | None = None,
              retrieval_relevant: bool | None = None) -> tuple[str, bool]:
    """Return (safe_answer, was_replaced)."""
    if _is_empty(answer):
        logger.info("Guard replaced empty response.")
        return SAFE_FALLBACK, True

    if _leaks_internals(answer):
        logger.info("Guard replaced response leaking internals.")
        return SAFE_FALLBACK, True

    if _claims_unfound_order(answer, order_number, order_found):
        logger.info("Guard replaced response inventing order %s.", order_number)
        return order_not_found_message(order_number), True

    # FAQ answers that were generated without relevant retrieval should fall back
    # rather than risk hallucinated policy.
    if intent == "FAQ" and retrieval_relevant is False:
        logger.info("Guard replaced FAQ answer with no relevant retrieval.")
        return NOT_FOUND_FALLBACK, True

    if intent == "OUT_OF_SCOPE":
        # Even if the LLM tried to answer, keep the controlled fallback.
        return OUT_OF_SCOPE_FALLBACK, True

    return answer, False