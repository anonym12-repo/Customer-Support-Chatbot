"""CrewAI Escalation Tool.

Creates a structured escalation record in data/escalations.json so a case can
be handed off to a human instead of the AI hallucinating a resolution.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from backend import config


class EscalationInput(BaseModel):
    reason: str = Field(..., description="Why the case is being escalated.")
    order_number: Optional[int] = Field(None, description="Related order number, if any.")
    priority: str = Field("high", description="Escalation priority: low, medium, high.")


def create_escalation(reason: str, order_number: int | None = None,
                      conversation_id: str = "", priority: str = "high") -> dict:
    record = {
        "id": f"esc_{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "conversation_id": conversation_id,
        "order_number": order_number,
        "reason": reason,
        "priority": priority,
        "status": "pending",
    }
    try:
        if config.ESCALATIONS_FILE.exists():
            existing = json.loads(config.ESCALATIONS_FILE.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        else:
            existing = []
        existing.append(record)
        config.ESCALATIONS_FILE.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    except Exception:
        pass
    return record


class EscalationTool(BaseTool):
    name: str = "create_escalation"
    description: str = (
        "Flag a customer case for human support. Use when the customer explicitly "
        "asks for a human, is highly frustrated, or reports a severe issue that "
        "cannot be resolved automatically."
    )
    args_schema: type[BaseModel] = EscalationInput

    def _run(self, reason: str, order_number: int | None = None,
             priority: str = "high") -> str:
        record = create_escalation(reason, order_number=order_number, priority=priority)
        return json.dumps({"escalated": True, "reference": record["id"]}, ensure_ascii=False)