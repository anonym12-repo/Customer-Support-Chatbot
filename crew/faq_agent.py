"""FAQ Specialist agent.

Answers company-policy questions strictly from retrieved FAQ content. It never
invents policy and clearly states when the FAQ does not contain the answer.
"""
from __future__ import annotations

from crewai import Agent

from backend import config
from backend.llm import get_llm
from backend.schemas import FAQHit
from tools.faq_tool import FAQRetrievalTool

from ._crew import run_single_task

_AGENT = None


def _get_agent() -> Agent:
    global _AGENT
    if _AGENT is not None:
        return _AGENT
    _AGENT = Agent(
        role="FAQ Specialist",
        goal=(
            "Answer company policy questions using ONLY the retrieved FAQ content. "
            "Never invent policies. If the FAQ does not contain the answer, say so."
        ),
        backstory=(
            "A meticulous support specialist who has memorised the company's official "
            "FAQ and refuses to speculate beyond it."
        ),
        llm=get_llm(),
        tools=[FAQRetrievalTool()],
        max_iter=3,  # bound LLM round-trips: answer directly from provided context
        verbose=False,
    )
    return _AGENT


def _load_prompt() -> str:
    return (config.PROMPTS_DIR / "faq_prompt.txt").read_text(encoding="utf-8")


def _format_context(hits: list[FAQHit]) -> str:
    if not hits:
        return "(No relevant FAQ content was retrieved.)"
    blocks = [
        f"[{h.faq_id}] (category: {h.category})\nQ: {h.question}\nA: {h.answer}"
        for h in hits
    ]
    return "\n\n".join(blocks)


def answer(question: str, hits: list[FAQHit]) -> tuple[str, float]:
    """Generate a grounded FAQ answer. Returns (answer, generation_ms)."""
    description = (
        f"{_load_prompt()}\n\n"
        f"FAQ CONTEXT (use ONLY this):\n{_format_context(hits)}\n\n"
        f"Customer question: {question}\n\n"
        f"Provide the answer and, on a new line, a source attribution of the form "
        f"'Source: Company FAQ — <category>'."
    )
    return run_single_task(_get_agent(), description)