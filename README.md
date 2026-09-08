# 🤖 Customer Support AI Chatbot

> **A production-oriented customer support assistant built with Python, CrewAI, ChromaDB, Ollama, and Streamlit — designed around deterministic routing, grounded RAG, conversational state, evaluation, latency measurement, and defensive AI engineering.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![CrewAI](https://img.shields.io/badge/CrewAI-Multi--Agent-orange)](https://www.crewai.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-RAG-purple)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black?logo=ollama)](https://ollama.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/Tests-Pytest-green?logo=pytest)](https://pytest.org/)

---

## 📌 Project Overview

This project is an **AI-powered e-commerce customer support system** that can answer company-policy questions, retrieve order information, maintain conversational context, and escalate requests when necessary.

The project was deliberately designed as a **Python-first AI pipeline rather than a simple LLM chatbot**.

Instead of sending every user message directly to an LLM, the system uses a **deterministic-first architecture**:

```text
                         ┌─────────────────────┐
                         │     Streamlit UI     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Conversation State │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Deterministic Router│
                         │ Intent + Order ID   │
                         └──────────┬──────────┘
                                    │
                ┌───────────────────┼───────────────────┐
                ▼                   ▼                   ▼
        ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
        │  Order Data  │    │  ChromaDB    │    │ Deterministic│
        │  O(1) Lookup │    │     RAG      │    │   Responses  │
        └──────┬───────┘    └──────┬───────┘    └──────────────┘
               │                   │
               └──────────┬────────┘
                          ▼
                 ┌─────────────────┐
                 │   CrewAI Layer  │
                 │ FAQ / Order /   │
                 │ Support Agents  │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Grounding Guard │
                 │ + Final Answer  │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Customer Reply  │
                 └─────────────────┘
```

### Why this architecture?

A common beginner AI project looks like:

```text
User → Prompt → LLM → Answer
```

This project intentionally goes further:

```text
User
  ↓
Intent detection
  ↓
Order/entity extraction
  ↓
Conversation-state resolution
  ↓
Deterministic data retrieval
  ↓
RAG when required
  ↓
LLM generation only when required
  ↓
Grounding / safety validation
  ↓
Final response
```

This reduces unnecessary LLM calls, improves latency, makes failures easier to diagnose, and reduces the opportunity for the model to invent business information.

---

# ✨ Key Features

## 💬 1. Customer Support Chat

The assistant supports natural-language questions such as:

- "What is your return policy?"
- "How long does shipping take?"
- "Where is my order 1004?"
- "What did I order?"
- "When will it arrive?"
- "Can I return this?"
- "Thanks for your help."

The application maintains recent conversation context so follow-up questions can refer to a previously mentioned order.

Example:

```text
User: Where is order 1004?

Bot: Order #1004 is currently Shipped...

User: When will it arrive?

Bot: [Resolves "it" against the active order context]
```

---

# 🧠 2. Deterministic-First AI Architecture

The system does **not** use an LLM for every operation.

Python handles predictable tasks first:

- Intent routing
- Order-number extraction
- Active-order resolution
- Order existence checks
- Deterministic responses
- Similarity thresholds
- Validation/guards
- Escalation record creation

The LLM is primarily used for **natural-language generation after the relevant data has already been gathered**.

### Benefits

- Lower latency
- Lower compute usage
- More predictable behavior
- Easier debugging
- Reduced hallucination surface
- Clear separation between deterministic logic and generative AI

This is one of the main engineering principles demonstrated by the project.

---

# 🔎 3. Retrieval-Augmented Generation (RAG)

Company policy information is stored in a FAQ dataset and indexed into **ChromaDB**.

The retrieval pipeline uses:

```text
Company FAQ
     ↓
PDF / structured FAQ source
     ↓
Text extraction
     ↓
Question + Answer chunks
     ↓
Ollama embeddings
     ↓
ChromaDB
     ↓
Similarity search
     ↓
Threshold filtering
     ↓
Grounded LLM response
```

The system does not blindly accept the top retrieved result.

A configurable similarity threshold is used so that weak matches can be rejected instead of being passed to the LLM as if they were authoritative.

### Why this matters

Without relevance filtering:

```text
User asks unrelated question
        ↓
Nearest FAQ is returned anyway
        ↓
LLM treats it as context
        ↓
Potentially misleading answer
```

With threshold filtering:

```text
User question
      ↓
Vector search
      ↓
Is the result relevant enough?
   ↙              ↘
 YES              NO
  ↓                ↓
Grounded answer   Safe decline
```

---

# ⚡ 4. Caching & Performance Optimization

Frequently accessed FAQ quick actions are handled through an **in-memory cache/dictionary**.

This avoids unnecessary:

- Vector database searches
- Embedding generation
- LLM calls

for predictable, predefined FAQ interactions.

For dictionary-backed quick actions, lookup is **O(1) average-case time complexity**.

The broader system also caches reusable resources such as data/index-related objects where appropriate.

### Performance instrumentation

The backend records stage-level timing including:

```text
routing_ms
retrieval_ms
tool_ms
generation_ms
total_ms
```

This makes it possible to identify whether a slow response is caused by:

- routing
- retrieval
- data access
- LLM generation
- another pipeline stage

rather than guessing about performance.

---

# 🤖 5. CrewAI Multi-Agent Architecture

CrewAI is used for the generative support layer.

The project contains three specialist roles:

### FAQ Agent

Responsible for company policies and FAQ-grounded responses.

### Order Agent

Responsible for retrieving and explaining order information.

### Support Agent

Acts as the customer-support response layer and coordinates the final user-facing response.

The important design choice is that agents are **not trusted as the database**.

They receive information gathered by deterministic tools and retrieval components.

```text
Customer Request
       ↓
Python Pipeline
       ↓
Relevant Data / RAG Context
       ↓
CrewAI Specialist
       ↓
Grounded Natural-Language Response
```

---

# 🗃️ 6. Deterministic Order Lookup

Order information is stored in CSV files for the demonstration environment.

Instead of semantic search, order numbers use deterministic matching.

This is intentional.

An order ID such as:

```text
1004
```

should not be retrieved using semantic similarity.

The system performs an exact lookup against cached order data.

This demonstrates an important engineering principle:

> **Use deterministic algorithms for deterministic problems and LLMs for tasks that actually require language reasoning.**

Product IDs are parsed using `ast.literal_eval` rather than unsafe `eval()`.

---

# 🧩 7. Conversational State

The chatbot maintains lightweight conversation state to support follow-up questions.

For example:

```text
User: Show me order 1004

Bot: Order #1004 is shipped.

User: What products are in it?

Bot: Order #1004 contains...
```

The system can resolve references such as:

- "it"
- "that order"
- "this order"

against the active order context.

Explicit order numbers take precedence over stored state.

This prevents a previous order from unintentionally hijacking a new request.

---

# 🛡️ 8. Defensive AI / Guardrails

The system includes multiple defensive layers.

### Examples

**No matching order**

```text
Order requested
      ↓
Exact lookup
      ↓
Not found
      ↓
Deterministic response
      ↓
No LLM call required
```

**Irrelevant FAQ question**

```text
Question
   ↓
ChromaDB retrieval
   ↓
Similarity below threshold
   ↓
Safe decline
```

**Unsupported policy request**

The assistant is instructed not to invent company policies when the knowledge base does not provide sufficient evidence.

### Security-conscious implementation

- `.env` is excluded from version control
- `.env.example` documents required configuration
- No credentials are stored in the repository
- `eval()` is avoided
- Order/product data is synthetic
- Internal errors are not intentionally exposed as raw stack traces to customers

---

# 🚨 9. Escalation Handling

Requests that require human intervention can be recorded through the escalation tool.

Escalation records are stored locally for the demonstration environment.

This creates a path from:

```text
AI Self-Service
      ↓
Unable / inappropriate for automation
      ↓
Escalation
      ↓
Human support workflow
```

A production implementation could replace the local JSON storage with a database, ticketing platform, CRM, or help-desk API.

---

# 📊 10. Evaluation & Benchmarking

This project includes a dedicated evaluation layer rather than relying only on manual testing.

```text
evaluation/
├── benchmark.py
├── evaluate.py
├── evaluation_dataset.csv
└── metrics.py
```

The evaluation dataset covers different support scenarios and intents.

The project can measure areas such as:

- Intent classification
- Order ID extraction
- Retrieval quality
- Response behavior
- Latency
- Per-stage latency
- Average latency
- P95 latency

The benchmark is designed to report **measured results rather than fabricated performance numbers**.

> Run the evaluation locally after installing dependencies and starting Ollama.


---

# 🖥️ User Interface

The chatbot is implemented with **Streamlit**.

The UI includes:

- Chat interface
- Conversation history
- FAQ quick-action buttons
- Clear conversation functionality
- Ollama health/status checks
- Model availability diagnostics
- Vector-index availability checks
- Optional developer diagnostics
- Stage-level latency information

### FAQ Quick Actions

Users can select frequently asked questions directly instead of typing every request.

This improves accessibility and also demonstrates the performance benefit of the deterministic FAQ cache.

---

# 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| **Python** | Core application and orchestration |
| **CrewAI** | Agent orchestration / specialist roles |
| **Ollama** | Local LLM + embeddings |
| **Llama 3.1 8B** | Local language model |
| **nomic-embed-text** | Embedding model |
| **ChromaDB** | Vector database for RAG |
| **Streamlit** | Interactive web interface |
| **Pandas** | Structured order/product data |
| **PyPDF** | FAQ document processing |
| **Pydantic** | Typed schemas / validation |
| **Pytest** | Automated testing |
| **Docker** | Containerization support |
| **JSON / CSV** | Local demonstration data |

---

# 📁 Project Structure

```text
Customer Support Chatbot/
│
├── app.py                         # Streamlit application
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Container configuration
├── .env.example                   # Environment variable template
├── README.md
│
├── backend/
│   ├── config.py                  # Application configuration
│   ├── data.py                    # Data loading / caching
│   ├── guards.py                  # Output validation / safety guards
│   ├── llm.py                     # LLM configuration
│   ├── pipeline.py                # Main orchestration pipeline
│   ├── response_generator.py      # Response generation
│   ├── router.py                  # Deterministic intent routing
│   ├── schemas.py                 # Typed data structures
│   └── state.py                   # Conversation state
│
├── crew/
│   ├── faq_agent.py               # FAQ specialist
│   ├── order_agent.py             # Order specialist
│   └── support_agent.py           # Support response agent
│
├── tools/
│   ├── faq_tool.py                # FAQ retrieval tool
│   ├── order_tool.py              # Order lookup tool
│   └── escalation_tool.py         # Escalation tool
│
├── retrieval/
│   ├── embeddings.py              # Embedding configuration
│   ├── index_faq.py               # Build ChromaDB index
│   └── retriever.py               # Similarity search / filtering
│
├── data/
│   ├── Company_FAQ.pdf            # FAQ knowledge source
│   ├── faqs.json                  # Structured FAQ source
│   ├── sample_orders.csv          # Synthetic order data
│   ├── sample_products.csv        # Synthetic product data
│   ├── escalations.json           # Local escalation records
│   └── build_faq_pdf.py
│
├── evaluation/
│   ├── benchmark.py               # Latency benchmarking
│   ├── evaluate.py                # Evaluation runner
│   ├── metrics.py                 # Evaluation metrics
│   └── evaluation_dataset.csv
│
├── tests/                         # Automated tests
├── prompts/                       # Prompt templates
├── chroma_db/                     # Local vector index
├── base44/                        # Development/build artefacts
└── src/                           # Supporting project files
```

> **Note:** `.env` and the local `venv/` should not be committed to GitHub. `chroma_db/` can also be excluded if you prefer to have users rebuild the index locally.

---

# 🚀 Getting Started

## 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd "Customer Support Chatbot"
```

## 2. Create a virtual environment

### Windows

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure environment variables

Copy:

```text
.env.example
```

to:

```text
.env
```

Configure the Ollama model/API settings as required by the project.

**Never commit your `.env` file or real credentials to GitHub.**

---

# 🦙 Ollama Setup

Install Ollama and make sure the Ollama service is running.

Pull the required models:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

Start the service if required:

```bash
ollama serve
```

---

# 📚 Build the FAQ Index

The vector index needs to be created before running retrieval-based FAQ queries.

```bash
python data/build_faq_pdf.py
python -m retrieval.index_faq
```

The index is stored locally in:

```text
chroma_db/
```

If the FAQ source changes, rebuild the index.

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal.

---

# 📈 Run Evaluation

```bash
python -m evaluation.evaluate
```

---

# ⏱️ Run Latency Benchmark

```bash
python -m evaluation.benchmark
```

For additional benchmark rounds:

```bash
python -m evaluation.benchmark --rounds 5
```

---

# 🎯 Engineering Skills Demonstrated

This project demonstrates more than simply knowing how to call an LLM API.

### AI / Machine Learning Engineering

- Retrieval-Augmented Generation (RAG)
- Semantic search
- Text embeddings
- Vector databases
- Local LLM inference
- Prompt engineering
- Multi-agent orchestration
- Grounded generation
- AI evaluation
- Hallucination mitigation

### Backend Engineering

- Python application architecture
- Modular system design
- Separation of concerns
- Deterministic routing
- Stateful conversation management
- Data-access abstraction
- Error handling
- Caching
- Complexity-aware design
- Dependency management

### Software Engineering

- Unit testing
- Integration-oriented testing
- Type/schema validation
- Defensive programming
- Configuration management
- Environment variables
- Git/GitHub project organization
- Documentation
- Containerization

### Performance Engineering

- LLM call minimization
- O(1) dictionary lookups
- Resource caching
- Stage-level latency instrumentation
- Average latency measurement
- P95 latency measurement
- Retrieval thresholding

### AI Reliability

- Grounded responses
- Deterministic fallbacks
- Retrieval relevance thresholds
- Unsupported-question handling
- Order-ID validation
- Conversation-state controls
- Output guards
- Human escalation path

---

# 💡 Important Design Decisions

## Why not use the LLM for everything?

LLMs are powerful but expensive and unpredictable compared with deterministic code.

For example, checking whether order `1004` exists does not require language reasoning.

Therefore:

```python
orders.get(order_id)
```

is preferable to:

```text
Ask an LLM to determine whether order 1004 exists.
```

This makes the application faster, cheaper, and more reliable.

---

## Why use RAG?

Company policies change and should come from an authoritative knowledge source.

Instead of asking the LLM to rely on its pretrained knowledge:

```text
Company FAQ → Retrieval → Relevant context → LLM
```

This makes the response generation grounded in the application's own knowledge base.

---

## Why use a similarity threshold?

A vector database will usually return the nearest available results, even when none are actually relevant.

The application therefore distinguishes:

```text
Nearest result
```

from:

```text
Relevant result
```

Only sufficiently relevant results are used for grounded FAQ generation.

---

## Why use CrewAI?

CrewAI provides a structured way to model specialist responsibilities.

The architecture separates:

```text
FAQ knowledge
Order knowledge
Customer-support response generation
```

while keeping deterministic data retrieval outside the agents.

This demonstrates practical multi-agent orchestration rather than using multiple agents simply for complexity's sake.

---

# 🔐 Security & Production Considerations

This repository uses synthetic demonstration data and local models.

A production deployment would require additional controls, including:

- Authentication and authorization
- Persistent database storage
- Rate limiting
- Audit logging
- PII detection/redaction
- Secrets management
- Role-based access control
- Secure customer identity verification before exposing order information
- Production monitoring/observability
- Distributed conversation state
- Persistent vector-store management
- API-level access controls
- Human support/ticketing integration

The current project intentionally focuses on demonstrating the **AI application architecture and engineering workflow** rather than claiming to be a fully production-hardened enterprise system.

---

# 🚧 Current Limitations

- Ollama currently provides local model inference, so response speed depends on local hardware.
- Conversation state is local/in-memory for the demonstration.
- Order/product data is synthetic CSV data.
- Escalations are stored locally rather than in a production ticketing system.
- The LLM can still produce imperfect language; grounding and guards reduce but cannot mathematically eliminate every possible generation error.
- The system currently targets a defined set of customer-support use cases rather than arbitrary customer service.

---

# 🔮 Future Improvements

Potential next steps include:

### Production Data Layer
Replace CSV/JSON storage with PostgreSQL or another production database.

### Customer Authentication
Require verification before exposing sensitive order information.

### Persistent Conversations
Store conversation state in Redis or a database.

### Enterprise Knowledge Base
Support multiple documents, policy versions, metadata filters, and automatic re-indexing.

### Observability
Integrate structured logging and monitoring with tools such as OpenTelemetry/Grafana.

### Human Handoff
Connect escalation directly to a help-desk/CRM system.

### API Layer
Expose the pipeline through FastAPI for integration with web/mobile applications.

### Advanced Evaluation
Add semantic answer evaluation, retrieval precision/recall, regression testing, and automated evaluation in CI/CD.

### Deployment
Deploy the Streamlit/API layer and model infrastructure as separate services.

---

# 📸 Demo

### Chatbot Interface

<img width="1911" height="867" alt="image" src="https://github.com/user-attachments/assets/758f679c-de8e-4a28-a2c5-2bfa792e58e8" />

```text
docs/
└── chatbot-screenshot.png
```

### Example Interaction

<img width="400" height="178" alt="20260908-2156-48 5427550" src="https://github.com/user-attachments/assets/ada25d2f-3220-411b-92c1-aa349e06ad5a" />


```text
docs/
└── demo.gif
```

Then embed it using:

```markdown
![Customer Support Chatbot Demo](docs/demo.gif)
```

A short demo should ideally show:

1. A FAQ question
2. A quick-action FAQ selection
3. An order lookup
4. A follow-up question using conversation context
5. A case where the system safely declines an unsupported question

---

# 📊 Example Supported Scenarios

| Scenario | Processing Strategy |
|---|---|
| FAQ / policy question | RAG + grounded generation |
| FAQ quick action | Cached deterministic lookup |
| Existing order lookup | Deterministic data lookup + LLM response |
| Follow-up order question | Conversation state + order lookup |
| Unknown order | Deterministic response |
| Irrelevant FAQ question | Retrieval threshold + safe decline |
| Greeting / thanks | Deterministic response |
| Escalation request | Escalation workflow |

---

# 🏗️ Architecture Philosophy

The central principle of this project is:

> **Do not use an LLM where deterministic software can solve the problem more reliably.**

The system therefore follows a hybrid architecture:

```text
              ┌─────────────────────────────┐
              │      Natural Language       │
              └──────────────┬──────────────┘
                             │
                             ▼
                   ┌──────────────────┐
                   │ Python Routing   │
                   └────────┬─────────┘
                            │
             ┌──────────────┼──────────────┐
             │              │              │
             ▼              ▼              ▼
        Exact Data        RAG Search    Deterministic
        Retrieval         ChromaDB       Response
             │              │              │
             └──────────────┼──────────────┘
                            ▼
                     ┌──────────────┐
                     │    CrewAI    │
                     │ LLM Response │
                     └──────┬───────┘
                            ▼
                     ┌──────────────┐
                     │    Guards    │
                     └──────┬───────┘
                            ▼
                     Customer Answer
```

This design makes the system easier to reason about, test, optimize, and extend.

---

# 👩‍💻 Why I Built This

This project was built to explore what it takes to move from a **basic chatbot prototype to an engineered AI application**.

The focus was not simply on generating good-looking responses, but on questions such as:

- When should an LLM be called?
- When should deterministic code take over?
- How can retrieval quality be controlled?
- How can hallucination risk be reduced?
- How can conversational context be maintained?
- How can latency be measured instead of guessed?
- How can an AI workflow be tested?
- How can the system fail safely?
- How would this architecture evolve toward production?

The resulting system combines **AI engineering, backend development, information retrieval, software testing, and performance engineering** in a single project.

---

# ⭐ Recruiter Snapshot

**This project demonstrates experience with:**

`Python` · `RAG` · `LLMs` · `CrewAI` · `ChromaDB` · `Ollama` · `Embeddings` · `Prompt Engineering` · `Multi-Agent Systems` · `Streamlit` · `Caching` · `State Management` · `Testing` · `Evaluation` · `Latency Benchmarking` · `Guardrails` · `Error Handling` · `Docker` · `Git/GitHub`

---


## ⚠️ Disclaimer

This project is a portfolio/educational implementation using synthetic customer, order, and product data. It should not be interpreted as a production-ready customer support system without the additional security, authentication, monitoring, persistence, and operational controls described above.
