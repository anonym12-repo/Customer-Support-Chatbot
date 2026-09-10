"""Order Specialist agent.

Answers order questions using structured order data provided by the deterministic
order lookup. It states facts only and never invents orders or statuses.
"""
from __future__ import annotations

from crewai import Agent

from backend import config
from backend.llm import get_llm
from backend.schemas import OrderInfo
from tools.order_tool import OrderLookupTool

from ._crew import run_single_task

_AGENT = None


def _get_agent() -> Agent:
    global _AGENT
    if _AGENT is not None:
        return _AGENT
    _AGENT = Agent(
        role="Order Specialist",
        goal=(
            "Answer questions about customer orders using the provided order data. "
            "State facts only. If an order was not found, say so clearly."
        ),
        backstory=(
            "A precise support specialist who works from the order database and "
            "never guesses about order status or contents."
        ),
        llm=get_llm(),
        tools=[OrderLookupTool()],
        max_iter=3,  # bound LLM round-trips: answer directly from provided context
        verbose=False,
    )
    return _AGENT


def _load_prompt() -> str:
    return (config.PROMPTS_DIR / "order_prompt.txt").read_text(encoding="utf-8")


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


def answer(question: str, order: OrderInfo) -> tuple[str, float]:
    """Generate a grounded order answer. Returns (answer, generation_ms)."""
    description = (
        f"{_load_prompt()}\n\n"
        f"ORDER DATA (facts only):\n{_format_order(order)}\n\n"
        f"Customer question: {question}\n\nAnswer:"
    )
    return run_single_task(_get_agent(), description)