"""Streamlit frontend for the AI Customer Support Assistant.

The UI is intentionally thin: it collects a message, calls the deterministic
Python pipeline, and renders the grounded response. Internal reasoning is
never exposed to the customer; a Developer Mode toggle shows safe diagnostics.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time

import streamlit as st

# Ensure the project root is importable when running `streamlit run app.py`
# from any working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend import config, data  # noqa: E402
from backend.pipeline import handle_message  # noqa: E402

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL, logging.INFO))

st.set_page_config(
    page_title="AI Customer Support Assistant",
    page_icon="🎧",
    layout="centered",
)

st.title("AI Customer Support Assistant")
st.caption("Get help with orders, shipping, returns, payments, and company policies.")


# --- Ollama health check (server reachable + required models installed) ---
@st.cache_data(show_spinner=False)
def ollama_status() -> dict:
    try:
        import urllib.request

        req = urllib.request.Request(
            f"{config.OLLAMA_API_BASE}/api/tags", headers={"Accept": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            tags = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return {"reachable": False, "missing": [config.OLLAMA_MODEL, config.EMBED_MODEL]}
    installed = {m.get("name", "") for m in tags.get("models", [])}
    missing = [m for m in (config.OLLAMA_MODEL, config.EMBED_MODEL) if m not in installed]
    return {"reachable": True, "missing": missing}


_ollama = ollama_status()
if not _ollama["reachable"]:
    st.error(
        "AI backend unavailable. Please make sure Ollama is running "
        f"at {config.OLLAMA_API_BASE} (e.g. `ollama serve`)."
    )
elif _ollama["missing"]:
    st.warning(
        "Ollama is running, but these required models are not installed: "
        f"**{', '.join(_ollama['missing'])}**. Install them with: "
        + ", ".join(f"`ollama pull {m}`" for m in _ollama["missing"])
    )

# --- FAQ index check (indexing is a separate, explicit step -- never automatic) ---
if not config.CHROMA_PATH.exists():
    st.warning(
        "FAQ vector index not found. FAQ questions will not be answerable "
        "until you run: `python -m retrieval.index_faq` "
        "(after `python data/build_faq_pdf.py`)."
    )


# --- Sidebar: developer mode + actions ---
with st.sidebar:
    st.header("Settings")
    dev_mode = st.toggle("Developer Mode", value=False)
    if st.button("Clear conversation"):
        st.session_state.clear()
        st.rerun()
    st.divider()
    st.subheader("About")
    st.write(
        "Deterministic-first AI support: routing, order lookup and validation "
        "run in Python; the local LLM only generates grounded answers."
    )
    if dev_mode:
        st.json(config.as_dict())


# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = None
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []


# --- FAQ quick actions (loaded dynamically from the FAQ dataset) ---
@st.cache_data(show_spinner=False)
def load_faq_questions() -> list[dict]:
    import json

    path = config.DATA_DIR / "faqs.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


faqs = load_faq_questions()
if faqs:
    st.subheader("FAQ Quick Actions")
    categories: dict[str, list[dict]] = {}
    for f in faqs:
        categories.setdefault(f["category"], []).append(f)
    cat_labels = {"shipping": "Shipping", "returns": "Returns",
                  "payments": "Payments", "orders": "Orders", "general": "General"}
    for cat, items in categories.items():
        with st.expander(cat_labels.get(cat, cat.title())):
            cols = st.columns(2)
            for idx, item in enumerate(items):
                with cols[idx % 2]:
                    if st.button(item["question"], key=f"faq_{item['faq_id']}",
                                 use_container_width=True):
                        st.session_state["pending_input"] = item["question"]

st.divider()


# --- Chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if dev_mode and msg.get("diagnostics"):
            with st.expander("Diagnostics"):
                st.json(msg["diagnostics"])


# --- Input handling ---
def submit(text: str) -> None:
    if not text.strip():
        return
    st.session_state.messages.append({"role": "user", "content": text})
    with st.chat_message("user"):
        st.markdown(text)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            t0 = time.perf_counter()
            try:
                result = handle_message(
                    text, st.session_state.conversation_id
                )
                st.session_state.conversation_id = result.state.conversation_id
                answer = result.answer
                diagnostics = None
                if dev_mode:
                    diagnostics = {
                        "intent": result.intent,
                        "retrieval_used": result.metrics.retrieval_used,
                        "faq_id": result.metrics.faq_id,
                        "retrieval_score": round(result.metrics.top_similarity, 4),
                        "routing_ms": round(result.metrics.routing_ms, 2),
                        "retrieval_ms": round(result.metrics.retrieval_ms, 2),
                        "tool_ms": round(result.metrics.tool_ms, 2),
                        "generation_ms": round(result.metrics.generation_ms, 2),
                        "total_ms": round(result.metrics.total_ms, 2),
                        "active_order": result.state.active_order_number,
                    }
                    if diagnostics:
                        with st.expander("Diagnostics"):
                            st.json(diagnostics)
                if result.sources:
                    src = result.sources[0]
                    st.caption(f"Source: Company FAQ — {src.get('category', 'policy')}")
            except Exception:
                answer = (
                    "Something went wrong on our side. Please try again or "
                    "contact customer support."
                )
                diagnostics = None
            elapsed = (time.perf_counter() - t0) * 1000
        st.markdown(answer)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "diagnostics": diagnostics if dev_mode else None,
        }
    )
    st.rerun()


pending = st.session_state.pop("pending_input", None)
if pending:
    submit(pending)

user_input = st.chat_input("Ask about an order or a company policy...")
if user_input:
    submit(user_input)