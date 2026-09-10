"""Customer Support Specialist agent.

Handles combined FAQ+order questions (general greetings are answered
deterministically in the pipeline and never reach an agent). The agent clearly
separates ORDER FACTS from COMPANY POLICY and never invents policy information.
"""
from __future__ import annotations

from crewai import Agent

from backend import config
from backend.llm import get_llm
from backend.schemas import FAQHit, OrderInfo
from tools.escalation_tool import EscalationTool
from tools.faq_tool import FAQRetrievalTool
from tools.order_tool import OrderLookupTool

from ._crew import run_single_task

_AGENT = None


def _get_agent() -> Agent:
    global _AGENT
    if _AGENT is not None:
        return _AGENT
    _AGENT = Agent(
        role="Customer Support Specialist",
        goal=(
            "Help customers with combined order and policy questions, general "
            "greetings, and escalations. Distinguish order facts from company "
            "policy and never invent information."
        ),
        backstory=(
            "An empathetic senior support specialist who combines order data with "
            "company policy to give accurate, grounded answers, and knows when to "
            "escalate to a human."
        ),
        llm=get_llm(),
        tools=[OrderLookupTool(), FAQRetrievalTool(), EscalationTool()],
        max_iter=3,  # bound LLM round-trips: answer directly from provided context
        verbose=False,
    )
    return _AGENT


def _load_prompt(name: str) -> str:
    return (config.PROMPTS_DIR / name).read_text(encoding="utf-8")


def _format_order(order: OrderInfo) -> str:
    if not order.found:
        return f"Order #{order.order_number} was NOT found in the database."
    products = ", ".join(f"{p.product_id}:{p.name}" for p in order.products) or "none"
    return (
        f"Order #{order.order_number} FOUND.\n"
        f"Customer: {order.customer_name}\n"
        f"Status: {order.status}\n"
        f"Order date: {order.order_date}\n"
        f"Total: ${order.total:.2f}\n"
        f"Products: {products}"
    )


def _format_faq(hits: list[FAQHit]) -> str:
    if not hits:
        return "(No relevant FAQ content was retrieved.)"
    return "\n\n".join(
        f"[{h.faq_id}] (category: {h.category})\nQ: {h.question}\nA: {h.answer}"
        for h in hits
    )


def answer_combined(question: str, order: OrderInfo,
                     hits: list[FAQHit]) -> tuple[str, float]:
    description = (
        f"{_load_prompt('support_prompt.txt')}\n\n"
        f"ORDER FACTS:\n{_format_order(order)}\n\n"
        f"COMPANY POLICY (FAQ CONTEXT):\n{_format_faq(hits)}\n\n"
        f"Customer question: {question}\n\nAnswer:"
    )
    return run_single_task(_get_agent(), description)