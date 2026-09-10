"""Deterministic-first intent router.

Routing, order-id extraction and keyword detection are done with Python rules.
An LLM classifier is available as a *fallback* for genuinely ambiguous messages,
but the default path never calls the LLM just to decide intent.
"""
from __future__ import annotations

import logging
import re
import time

from . import data
from .schemas import RouterResult

logger = logging.getLogger(__name__)

# 4-digit order numbers as used in the demo dataset (1001-1005).
ORDER_RE = re.compile(r"\b(\d{4})\b")

FAQ_KEYWORDS = {
    "shipping", "ship", "delivery", "deliver", "return", "refund", "exchange",
    "payment", "pay", "cancel", "cancellation", "policy", "policies",
    "track", "tracking", "warranty", "international", "methods", "fee", "fees",
    "replace", "replacement", "damaged", "lost package", "backorder",
}

GREETING_WORDS = {"hi", "hello", "hey", "greetings", "howdy", "yo", "sup"}
GREETING_RE = re.compile(r"^\s*(hi|hello|hey|greetings|howdy|yo|sup|good (morning|afternoon|evening))\b", re.I)

# Conversational (non-policy) messages that are still in scope -> general help.
GENERAL_KEYWORDS = {"thank", "thanks", "thank you", "help", "please", "appreciate"}

ESCALATION_KEYWORDS = {
    "speak to a human", "talk to a human", "talk to someone", "speak to someone",
    "talk to an agent", "speak to an agent", "human agent", "real person",
    "customer service", "manager", "escalate", "escalation", "supervisor",
    "frustrated", "frustrating", "angry", "outrageous", "unacceptable",
    "extremely late", "never arrived", "never came", "still not here",
    "this is ridiculous", "complaint", "lawsuit", "sue",
}

# Topics the assistant is built for. If none match and the message is a clear
# question, it is treated as out-of-scope.
COMPANY_TOPIC_RE = re.compile(
    r"\b(order|orders|shipping|delivery|return|refund|exchange|payment|cancel|"
    r"policy|policies|track|tracking|warranty|product|products|package|"
    r"parcel|invoice|receipt|account|my order|status)\b",
    re.I,
)

# Conversational follow-ups that refer to the previously discussed order
# (anaphora such as "When will it arrive?"). Only consulted when the current
# message contains no order number of its own and a conversation-scoped
# active order is supplied by the caller.
FOLLOWUP_ORDER_RE = re.compile(
    r"\b(it|them|that order|this order|the order|my order|same order|"
    r"order status)\b",
    re.I,
)


def extract_order_number(text: str) -> int | None:
    """Extract a candidate 4-digit order number from the message (Python only)."""
    matches = ORDER_RE.findall(text or "")
    for m in matches:
        num = int(m)
        # Prefer numbers that actually exist in the dataset.
        if data.order_exists(num):
            return num
    # Fall back to the first 4-digit number even if unknown, so we can report
    # "order not found" rather than silently ignoring it.
    if matches:
        return int(matches[0])
    return None


def _contains_faq_intent(text: str) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in FAQ_KEYWORDS)


def _is_greeting(text: str) -> bool:
    stripped = (text or "").strip().lower()
    if len(stripped) < 30 and GREETING_RE.match(stripped):
        return True
    return stripped in GREETING_WORDS


def _is_escalation(text: str) -> bool:
    lowered = (text or "").lower()
    return any(kw in lowered for kw in ESCALATION_KEYWORDS)


def _is_general_conversational(text: str) -> bool:
    """Short conversational messages (thanks, help requests) -> general support."""
    lowered = (text or "").lower().strip()
    if len(lowered) > 60:
        return False
    return any(kw in lowered for kw in GENERAL_KEYWORDS)


def route(text: str, *, active_order: int | None = None,
          use_llm_fallback: bool = False) -> RouterResult:
    """Route a user message to a single intent using deterministic rules.

    ``active_order`` is the order number discussed earlier in the conversation
    (if any). It is used to resolve follow-up references like "When will it
    arrive?" without any LLM call.
    """
    start = time.perf_counter()
    text = text or ""

    order_number = extract_order_number(text)
    if (order_number is None and active_order is not None
            and FOLLOWUP_ORDER_RE.search(text)):
        # Follow-up message referring to the previously active order.
        order_number = int(active_order)
    has_faq = _contains_faq_intent(text)

    # Escalation takes priority when explicit human-handoff language is present.
    if _is_escalation(text):
        intent = "ESCALATION"
    elif order_number is not None and has_faq:
        intent = "FAQ_ORDER"
    elif order_number is not None:
        intent = "ORDER"
    elif has_faq:
        intent = "FAQ"
    elif _is_greeting(text):
        intent = "GENERAL"
    elif _is_general_conversational(text):
        intent = "GENERAL"
    elif COMPANY_TOPIC_RE.search(text):
        # Mentions a company topic but no specific keyword -> treat as general
        # support question rather than out-of-scope.
        intent = "GENERAL"
    else:
        # Ambiguous: optionally use LLM, otherwise mark out-of-scope.
        if use_llm_fallback:
            return _llm_route(text, order_number, start)
        intent = "OUT_OF_SCOPE"

    confidence = 0.98 if intent != "OUT_OF_SCOPE" else 0.7
    elapsed = (time.perf_counter() - start) * 1000
    return RouterResult(
        intent=intent,
        order_number=order_number,
        confidence=confidence,
        method="rules",
        raw={"text": text, "routing_ms": elapsed},
    )


def _llm_route(text: str, order_number: int | None, start: float) -> RouterResult:
    """LLM fallback classifier for ambiguous messages only."""
    try:
        from .llm import get_llm
        llm = get_llm()
        prompt = (
            "Classify the user message into exactly one of: FAQ, ORDER, "
            "FAQ_ORDER, GENERAL, OUT_OF_SCOPE, ESCALATION.\n"
            "Reply with the label only.\n\n"
            f"Message: {text}"
        )
        label = str(llm.call(prompt)).strip().upper()
        valid = {"FAQ", "ORDER", "FAQ_ORDER", "GENERAL", "OUT_OF_SCOPE", "ESCALATION"}
        intent = label if label in valid else "OUT_OF_SCOPE"
    except Exception as exc:
        logger.warning("LLM router fallback failed: %s", exc)
        intent = "OUT_OF_SCOPE"
    elapsed = (time.perf_counter() - start) * 1000
    return RouterResult(
        intent=intent,
        order_number=order_number,
        confidence=0.6,
        method="llm",
        raw={"text": text, "routing_ms": elapsed},
    )