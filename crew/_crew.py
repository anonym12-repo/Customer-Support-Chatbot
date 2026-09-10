"""Shared helper to run a single-agent, single-task CrewAI crew.

Keeps CrewAI usage in one place so it is not duplicated across agent files.
"""
from __future__ import annotations

import logging
import time

from crewai import Agent, Crew, Process, Task

from backend.llm import get_llm

logger = logging.getLogger(__name__)


def run_single_task(agent: Agent, description: str) -> tuple[str, float]:
    """Run one task for one agent and return (result_text, generation_ms)."""
    start = time.perf_counter()
    task = Task(
        description=description,
        expected_output="A concise, grounded customer-facing answer.",
        agent=agent,
    )
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )
    result = crew.kickoff()
    return str(result).strip(), (time.perf_counter() - start) * 1000