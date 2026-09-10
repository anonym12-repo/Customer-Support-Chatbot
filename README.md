# AI Customer Support Automation System

An AI customer-support automation system combining **retrieval-augmented generation (RAG)**, **deterministic business logic**, **tool calling**, **local LLM inference**, and **evaluation**.

This is an AI engineering portfolio project — not a simple chatbot. It demonstrates Python backend engineering, RAG with a local vector database, local LLMs via Ollama, CrewAI tool calling, deterministic routing, state management, error handling, latency optimization, evaluation, testing, and observability.

> **Engineering principle:** *LLMs should handle language and reasoning where necessary, while deterministic software should handle data access, validation, routing, business rules, and safety wherever possible.* The system is built **around** an LLM, not as a chatbot that asks an LLM everything.

---

## 1. Project Overview

A customer support assistant for a fictional e-commerce company that can:

- Answer policy questions using RAG over a company FAQ PDF.
- Look up customer orders from structured data.
- Understand conversational references to previously mentioned orders.
- Handle questions involving both policy and a specific order.
- Detect out-of-scope questions and refuse gracefully.
- Escalate appropriate cases to a human instead of hallucinating.
- Maintain conversation state.
- Provide fast responses by avoiding unnecessary LLM calls.
- Record latency and system metrics, with automated evaluation and tests.

## 2. Problem Statement

Naive "send everything to the LLM" support bots are slow, expensive, and prone to hallucination: they may invent order statuses, fabricate policies, or leak internal reasoning. This project solves that by separating **deterministic logic** (routing, data access, validation, safety) from **language generation**, so the LLM is only used where it adds value.

## 3. Architecture

```mermaid
flowchart TD
    U[User] --> UI[Streamlit UI]
    UI --> APP[Python application layer]
    APP --> ROUTER[Intent + entity routing\n(deterministic Python)]
    ROUTER --> WF{Specialized workflow}
    WF --> FAQ[FAQ workflow\nRAG over ChromaDB]
    WF --> ORDER[Order workflow\npandas lookup]
    WF --> COMBO[FAQ + Order workflow\nretrieval + lookup]
    WF --> GEN[General support workflow]
    WF --> OOS[Out-of-scope workflow\ncontrolled fallback]
    WF --> ESC[Escalation workflow\nhuman handoff]
    FAQ --> GEN2[Grounded response generation\nCrewAI + Ollama]
    ORDER --> GEN2
    COMBO --> GEN2
    GEN --> GUARD[Response validation / safety guard]
    GEN2 --> GUARD
    ESC --> GUARD
    OOS --> GUARD
    GUARD --> FINAL[Final response]
    FINAL --> OBS[Observability / metrics]
```

### Data flow

1. The user message enters the **router** (Python rules): extract order IDs (including resolving follow-ups like "When will it arrive?" to the previously discussed order), detect FAQ keywords, greetings, escalation language, or out-of-scope.
2. The router selects a single **workflow**.
3. Deterministic data is gathered **before** any LLM call:
   - FAQ: embed the query and retrieve from ChromaDB with a similarity threshold.
   - Order: O(1) lookup from a cached pandas index.
   - Combined: both.
4. The gathered context is passed to the relevant **CrewAI agent**, which makes **one** LLM call to phrase a grounded answer. If the order was not found or no FAQ hit passed the relevance threshold, the system declines **deterministically instead** — those paths make zero LLM calls.
5. A **response guard** validates the LLM-written answer deterministically (empty responses, leaked internals, unfound-order claims, out-of-scope leaks). Composed deterministic answers need no validation by construction.
6. The final answer and structured metrics are returned to the UI and logged.

## 4. Technology Stack

| Layer | Technology |
|------|-----------|
| Frontend | Streamlit |
| Application / routing / state / guards | Python |
| Agents / tool calling | CrewAI |
| Vector database | ChromaDB |
| Embeddings & LLM | Ollama (`nomic-embed-text`, `llama3.1:8b`) |
| Structured data | pandas |
| FAQ ingestion | PyPDF |
| Configuration | python-dotenv |
| Testing | pytest |

## 5. Why Deterministic Routing?

Routing with Python rules is **fast** (sub-millisecond), **reproducible**, and **cheap**. Using an LLM to decide whether a message contains a 4-digit order ID would add latency and cost for zero reliability gain. The router only falls back to an LLM classifier for genuinely ambiguous messages, and even then only when enabled.

## 6. Why RAG?

Company policy lives in a FAQ PDF. RAG grounds answers in the authoritative source and provides **source attribution**, dramatically reducing hallucination versus asking the LLM from parametric memory.

## 7. Why ChromaDB?

ChromaDB is a lightweight, persistent, local vector store that pairs cleanly with Ollama embeddings. It keeps the entire system running locally with no third-party hosted vector database, which matters for privacy, cost, and reproducibility.

## 8. Why Ollama?

Ollama provides **local** LLM and embedding inference. No data leaves the machine, there are no API keys or per-token costs, and the system is fully reproducible after cloning. Default models: `llama3.1:8b` (generation) and `nomic-embed-text` (embeddings).

## 9. Why CrewAI?

CrewAI gives a clean abstraction for specialized agents with typed tools. We use it where it adds value — generating grounded answers from gathered context — while deliberately **not** routing every message through multiple agents. Each request triggers a single-agent, single-task crew (one LLM call).

## 10. Project Structure

```
customer-support-ai/
├── app.py                  # Streamlit frontend
├── backend/
│   ├── config.py           # Central configuration (env-driven)
│   ├── llm.py              # Cached CrewAI/Ollama LLM instance
│   ├── schemas.py          # Typed dataclasses (shared structures)
│   ├── state.py            # Conversation state management
│   ├── data.py             # Cached pandas order/product lookup
│   ├── router.py           # Deterministic intent router
│   ├── response_generator.py  # Dispatches to CrewAI agents
│   ├── guards.py           # Deterministic response validation
│   └── pipeline.py         # End-to-end orchestrator (single source of truth)
├── crew/                   # CrewAI agents (single source of truth for CrewAI)
│   ├── faq_agent.py
│   ├── order_agent.py
│   └── support_agent.py
├── tools/                  # CrewAI tools (typed, deterministic)
│   ├── order_tool.py
│   ├── faq_tool.py
│   └── escalation_tool.py
├── retrieval/
│   ├── embeddings.py       # Ollama embedding function (cached)
│   ├── retriever.py        # Threshold-based ChromaDB retrieval
│   └── index_faq.py        # PDF ingestion + indexing
├── data/
│   ├── faqs.json           # FAQ source of truth
│   ├── build_faq_pdf.py    # Generates Company_FAQ.pdf
│   ├── sample_orders.csv
│   ├── sample_products.csv
│   └── escalations.json    # Runtime escalation records
├── prompts/                # Grounding prompt templates
├── evaluation/
│   ├── evaluation_dataset.csv
│   ├── metrics.py
│   ├── evaluate.py
│   └── benchmark.py
├── tests/                  # pytest suite
├── chroma_db/              # Persistent vector index (gitignored)
├── .env.example
├── requirements.txt
├── Dockerfile
└── README.md
```

> Note: CrewAI agents live in `crew/` (the platform reserves an `agents/` path). This is the single source of truth for CrewAI — the generation logic is not duplicated across files.

## 11. Installation

```bash
git clone <your-repo-url>
cd customer-support-ai
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env           # adjust if needed
```

## 12. Ollama Setup

Install [Ollama](https://ollama.com), then pull the models and start the server:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
ollama serve                   # runs on http://localhost:11434
```

The app performs an Ollama health check at startup and shows a clear message if the backend is unavailable.

## 13. FAQ Indexing

The FAQ PDF is generated from `data/faqs.json`, then indexed into ChromaDB:

```bash
python data/build_faq_pdf.py     # creates data/Company_FAQ.pdf
python -m retrieval.index_faq    # builds the ChromaDB collection
```

This is a **one-time** step. The index persists in `chroma_db/` and is not rebuilt during normal operation — normal chat traffic never touches the index build path. Re-run both steps whenever you change `data/faqs.json`.

## 14. Running the Application

```bash
streamlit run app.py
```

Then open the printed local URL. Use the **FAQ Quick Actions** buttons or type your own question. Toggle **Developer Mode** in the sidebar to see safe diagnostics (intent, retrieval score, per-stage latency).

## 15. Testing

```bash
pytest -v
```

Deterministic tests (router, orders, guards, out-of-scope, escalation) run without Ollama. Retrieval and LLM-dependent end-to-end tests **skip automatically** when Ollama or the ChromaDB index is unavailable.

## 16. Evaluation

```bash
python -m evaluation.evaluate
```

Measures (from real predictions, never fabricated):

- Intent accuracy
- Order extraction accuracy
- Order lookup accuracy
- FAQ retrieval top-1 accuracy
- FAQ retrieval top-3 recall
- Average routing latency
- Average retrieval latency

FAQ retrieval metrics require the index to be built; otherwise that section reports "n/a (index unavailable)".

## 17. Performance Measurement

Every request records structured latency metrics:

```json
{
  "routing_ms": 5,
  "retrieval_ms": 120,
  "tool_ms": 3,
  "generation_ms": 1400,
  "total_ms": 1528
}
```

Run the end-to-end latency benchmark (requires Ollama, plus a built FAQ index for the FAQ prompts):

```bash
python -m evaluation.benchmark            # add --rounds 5 for more samples
```

Optimizations implemented:

- Cached CSV loading (loaded once, O(1) order lookup).
- Cached ChromaDB client and collection.
- Cached, reusable LLM and embedding objects.
- Deterministic routing **before** any LLM invocation.
- At most **one** LLM call per request.
- No repeated CSV loading; no vector index rebuilds during normal execution.

## 18. Security Considerations

- **No `eval()`** — product ID lists are parsed with `ast.literal_eval()`.
- Order numbers are validated and coerced to integers.
- The response guard strips internal-reasoning leaks (chain-of-thought, tool/agent mentions, internal system names).
- `.env` is gitignored; secrets are never logged.
- Stack traces are never shown to end users — a friendly error is displayed and the technical error is logged server-side.
- All data is **synthetic** demo data.

## 19. Limitations

- LLM quality depends on the local model (`llama3.1:8b`); larger models improve nuance.
- The in-memory conversation store is not persistent across restarts.
- Escalation records are written to a local JSON file (no real email/CRM integration).
- RAG is limited to the provided FAQ PDF.
- No authentication/authorization — the demo is open.

## 20. Future Improvements

- Persistent conversation store (Redis/Postgres).
- Authentication, authorization, PII redaction, and audit logging for real customer data.
- Real escalation integration (email/CRM/ticketing).
- Hybrid retrieval (keyword + vector) and query rewriting.
- Streaming LLM responses for lower time-to-first-token.
- A/B evaluation across models and prompt variants.

---

## Engineering Decisions

### Why the application does NOT send every question through an LLM

Sending every message through one or more LLM agents adds latency, cost, and hallucination risk for tasks that are trivially deterministic. This system uses the LLM **only** for natural-language generation, after Python has already:

- Extracted the order ID (regex).
- Looked up the order (cached pandas index).
- Retrieved and threshold-filtered FAQ hits (ChromaDB).
- Routed to the correct workflow (keyword + pattern rules).

### How deterministic business logic reduces latency and hallucination risk

- **Latency:** routing is sub-millisecond; order lookup is O(1); only one LLM call is made, and only when needed. Out-of-scope, escalation, greeting, and order-not-found paths need **zero** LLM calls.
- **Hallucination:** the LLM is given explicit, structured context (order facts + retrieved FAQ) and strict grounding instructions; the response guard then deterministically blocks unfound-order claims, internal leaks, and ungrounded policy. The model is never asked to "remember" policies or invent order data.

---

## Docker

Ollama runs as a **separate service** because the LLM is local and not bundled in the app container:

```
Browser → Streamlit container → Ollama service → llama3.1:8b / nomic-embed-text
```

```bash
docker build -t customer-support-ai .
docker run -p 8501:8501 \
  -e OLLAMA_API_BASE=http://host.docker.internal:11434 \
  customer-support-ai
```

Ensure Ollama is running on the host and the models are pulled before starting the container.

---

## Data Privacy

All current data is **synthetic demonstration data** — no real personal information is used. The architecture is designed so that, in production, sensitive customer data could later be protected with authentication, authorization, encryption, audit logging, database access controls, and PII redaction. These production controls are **not** claimed to exist in this demo unless explicitly implemented.