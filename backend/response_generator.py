"""Grounded response generation dispatcher.

This module decides *which* CrewAI agent generates the answer. The actual CrewAI
implementation lives in the ``crew`` package (single source of truth). Each
request triggers exactly one LLM call.
"""
from __future__ import annotations

import logging

from backend.schemas import FAQHit, OrderInfo

logger = logging.getLogger(__name__)


def generate_faq_answer(question: str, hits: list[FAQHit]) -> tuple[str, float]:
    """Delegate to the FAQ Specialist agent."""
    from crew import faq_agent
    return faq_agent.answer(question, hits)


def generate_order_answer(question: str, order: OrderInfo) -> tuple[str, float]:
    """Delegate to the Order Specialist agent."""
    from crew import order_agent
    return order_agent.answer(question, order)


def generate_combined_answer(question: str, order: OrderInfo,
                               hits: list[FAQHit]) -> tuple[str, float]:
    """Delegate to the Support Specialist agent for FAQ+order questions."""
    from crew import support_agent
    return support_agent.answer_combined(question, order, hits)