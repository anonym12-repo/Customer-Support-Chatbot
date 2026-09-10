"""Typed data structures shared across the pipeline.

Using plain dataclasses keeps the deterministic parts of the system fast and
explicit, and gives the LLM layer structured context to ground its answers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RouterResult:
    intent: str  # FAQ | ORDER | FAQ_ORDER | GENERAL | OUT_OF_SCOPE | ESCALATION
    order_number: Optional[int] = None
    confidence: float = 1.0
    method: str = "rules"  # "rules" | "llm"
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProductInfo:
    product_id: int
    name: str


@dataclass
class OrderInfo:
    found: bool
    order_number: int
    customer_name: Optional[str] = None
    status: Optional[str] = None
    order_date: Optional[str] = None
    total: Optional[float] = None
    products: list[ProductInfo] = field(default_factory=list)


@dataclass
class FAQHit:
    faq_id: str
    question: str
    answer: str
    category: str
    source: str
    similarity: float


@dataclass
class RetrievalResult:
    relevant: bool
    hits: list[FAQHit] = field(default_factory=list)
    top_similarity: float = 0.0
    latency_ms: float = 0.0


@dataclass
class ConversationState:
    conversation_id: str
    active_order_number: Optional[int] = None
    last_intent: Optional[str] = None
    last_retrieved_faq: Optional[str] = None
    history: list[dict[str, str]] = field(default_factory=list)

    def update_from_router(self, router: RouterResult) -> None:
        if router.order_number is not None:
            self.active_order_number = router.order_number
        self.last_intent = router.intent


@dataclass
class Metrics:
    """Per-request latency / observability record (never fabricated)."""
    conversation_id: str = ""
    intent: str = ""
    order_number: Optional[int] = None
    retrieval_used: bool = False
    top_similarity: float = 0.0
    faq_id: Optional[str] = None
    tool_used: Optional[str] = None
    routing_ms: float = 0.0
    retrieval_ms: float = 0.0
    tool_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "intent": self.intent,
            "order_number": self.order_number,
            "retrieval_used": self.retrieval_used,
            "top_similarity": round(self.top_similarity, 4),
            "faq_id": self.faq_id,
            "tool_used": self.tool_used,
            "routing_ms": round(self.routing_ms, 2),
            "retrieval_ms": round(self.retrieval_ms, 2),
            "tool_ms": round(self.tool_ms, 2),
            "generation_ms": round(self.generation_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "success": self.success,
            "error": self.error,
        }


@dataclass
class PipelineResult:
    """The final output of handling one user message."""
    answer: str
    intent: str
    state: ConversationState
    metrics: Metrics
    sources: list[dict[str, str]] = field(default_factory=list)
    escalated: bool = False