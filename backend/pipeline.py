"""End-to-end pipeline orchestrator.

Single source of truth for handling one user message:
route -> resolve state -> gather deterministic data -> one LLM call -> guard.

The LLM is invoked at most once per request, and only for generation.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone

from . import config, data, guards, response_generator, router, state
from .schemas import Metrics, PipelineResult, RetrievalResult

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_escalation(order_number: int | None, reason: str,
                       conversation_id: str, priority: str = "high") -> dict:
    """Append an escalation record to data/escalations.json."""
    record = {
        "id": f"esc_{uuid.uuid4().hex[:8]}",
        "timestamp": _now_iso(),
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
        config.ESCALATIONS_FILE.write_text(
            json.dumps(existing, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        logger.error("Failed to write escalation record: %s", exc)
    return record

_faq_cache: dict[str, str] = {}

def _get_exact_faq(text: str) -> str | None:
    """O(1) exact-match cache for predefined UI buttons."""
    global _faq_cache
    if not _faq_cache:
        try:
            faq_path = config.DATA_DIR / "faqs.json"
            faqs = json.loads(faq_path.read_text(encoding="utf-8"))
            for item in faqs:
                _faq_cache[item["question"].strip().lower()] = item["answer"]
        except Exception as exc:
            logger.error("Failed to load FAQ cache: %s", exc)
            
    return _faq_cache.get(text.strip().lower())

def handle_message(text: str, conversation_id: str | None = None,
                   *, use_llm_fallback: bool = False) -> PipelineResult:
    """Process a single user message end-to-end."""
    total_start = time.perf_counter()
    conv_state = state.get_state(conversation_id) if conversation_id else state.new_conversation()
    metrics = Metrics(conversation_id=conv_state.conversation_id)
    _last_faq_category: str | None = None

    try:
        # 1. Route (deterministic).
        r_start = time.perf_counter()
        result = router.route(text, active_order=conv_state.active_order_number,
                              use_llm_fallback=use_llm_fallback)
        metrics.routing_ms = (time.perf_counter() - r_start) * 1000
        metrics.intent = result.intent
        metrics.order_number = result.order_number

        intent = result.intent

        # 2. Dispatch to the appropriate workflow.
        if intent == "OUT_OF_SCOPE":
            answer = guards.OUT_OF_SCOPE_FALLBACK
            metrics.success = True

        elif intent == "GENERAL":
            # Greetings / general help requests are answered deterministically
            # to avoid an unnecessary LLM call and stay robust when Ollama is down.
            answer = (
                "Hello! I can help with orders, shipping, returns, payments, and "
                "company policies. What would you like help with?"
            )

        elif intent == "ESCALATION":
            order_info = None
            if result.order_number is not None:
                t_start = time.perf_counter()
                order_info = data.get_order(result.order_number)
                metrics.tool_ms = (time.perf_counter() - t_start) * 1000
                metrics.tool_used = "order_lookup"
            reason = text
            esc = _append_escalation(
                result.order_number, reason, conv_state.conversation_id
            )
            metrics.tool_used = "escalation"
            answer = (
                f"I'm sorry you're having trouble. I've flagged your case for our "
                f"human support team (reference {esc['id']}). "
            )
            if order_info is not None and order_info.found:
                answer += (
                    f"I can see order #{order_info.order_number} is currently "
                    f"marked as {order_info.status}. "
                )
            answer += "A team member will reach out to you shortly."

        elif intent == "ORDER":
            t_start = time.perf_counter()
            order_info = data.get_order(result.order_number)
            metrics.tool_ms = (time.perf_counter() - t_start) * 1000
            metrics.tool_used = "order_lookup"
            
            if not order_info.found:
                # Deterministic decline: no LLM call is needed to say the
                # order does not exist, and none can fabricate details.
                answer = guards.order_not_found_message(result.order_number)
            else:
                # ZERO-LLM FAST PATH: Build the string directly in Python
                product_names = ", ".join(p.name for p in order_info.products)
                answer = (
                    f"Order #{order_info.order_number} is currently {order_info.status}. "
                    f"It was placed on {order_info.order_date} for a total of ${order_info.total:.2f}. "
                    f"The items included are: {product_names}."
                )
                metrics.generation_ms = 0.0  # Completely bypass the LLM
                
                answer, replaced = guards.validate(
                    answer, intent=intent,
                    order_number=result.order_number,
                    order_found=True,
                )

        elif intent == "FAQ":
            # ZERO-LLM FAST PATH: Check the exact-match cache first
            exact_answer = _get_exact_faq(text)
            
            if exact_answer:
                answer = exact_answer
                metrics.generation_ms = 0.0  # Bypass LLM completely
                metrics.retrieval_used = False
                metrics.tool_used = "faq_cache"
            else:
                from retrieval.retriever import retrieve_faq
                retrieval = retrieve_faq(text)
                metrics.retrieval_used = retrieval.relevant
                metrics.retrieval_ms = retrieval.latency_ms
                metrics.top_similarity = retrieval.top_similarity
                if retrieval.hits:
                    metrics.faq_id = retrieval.hits[0].faq_id
                    _last_faq_category = retrieval.hits[0].category
                if not retrieval.relevant:
                    answer = guards.NOT_FOUND_FALLBACK
                else:
                    answer, gen_ms = response_generator.generate_faq_answer(
                        text, retrieval.hits
                    )
                    metrics.generation_ms = gen_ms
                    answer, _ = guards.validate(
                        answer, intent=intent, retrieval_relevant=True
                    )
                    
        elif intent == "FAQ_ORDER":
            from retrieval.retriever import retrieve_faq
            t_start = time.perf_counter()
            order_info = data.get_order(result.order_number)
            metrics.tool_ms = (time.perf_counter() - t_start) * 1000
            metrics.tool_used = "order_lookup"
            retrieval = retrieve_faq(text)
            metrics.retrieval_used = retrieval.relevant
            metrics.retrieval_ms = retrieval.latency_ms
            metrics.top_similarity = retrieval.top_similarity
            if retrieval.hits:
                metrics.faq_id = retrieval.hits[0].faq_id
                _last_faq_category = retrieval.hits[0].category
            if not order_info.found:
                # Deterministic decline (plus the verbatim policy text if the
                # retrieval was relevant) -- zero LLM calls, zero fabrication.
                answer = guards.order_not_found_message(result.order_number)
                if retrieval.relevant and retrieval.hits:
                    answer += (
                        " For reference, here is the relevant policy: "
                        f"{retrieval.hits[0].answer}"
                    )
            elif not retrieval.relevant:
                # Order found, but no policy passed the threshold: answer the
                # order part only, with no policy claim.
                answer, gen_ms = response_generator.generate_order_answer(text, order_info)
                metrics.generation_ms = gen_ms
                answer, _ = guards.validate(
                    answer, intent=intent,
                    order_number=result.order_number,
                    order_found=True,
                    retrieval_relevant=False,
                )
            else:
                answer, gen_ms = response_generator.generate_combined_answer(
                    text, order_info, retrieval.hits
                )
                metrics.generation_ms = gen_ms
                answer, _ = guards.validate(
                    answer, intent=intent,
                    order_number=result.order_number,
                    order_found=True,
                    retrieval_relevant=True,
                )
        else:
            answer = guards.SAFE_FALLBACK

        # 3. Update conversation state.
        state.update_state(conv_state, result, answer)

    except Exception as exc:
        logger.exception("Pipeline failure handling message: %s", text)
        metrics.success = False
        metrics.error = str(exc)
        answer = guards.SAFE_FALLBACK

    metrics.total_ms = (time.perf_counter() - total_start) * 1000
    logger.info("request %s", json.dumps(metrics.to_log_dict()))

    sources = []
    if metrics.faq_id:
        category = _last_faq_category
        sources.append({
            "faq_id": metrics.faq_id,
            "category": category or "policy",
            "source": "Company_FAQ.pdf",
        })

    return PipelineResult(
        answer=answer,
        intent=metrics.intent,
        state=conv_state,
        metrics=metrics,
        sources=sources,
        escalated=metrics.tool_used == "escalation",
    )