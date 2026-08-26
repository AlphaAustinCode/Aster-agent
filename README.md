# 🛡️ Aster & Row AI Support Agent

A **secure, production-grade Retrieval-Augmented Generation (RAG) customer support agent** built for Aster & Row.

The system combines **grounded knowledge retrieval, secure order lookup, LLM guardrails, PII protection, isolated sessions, observability, and deterministic evaluation** to prevent hallucinations and data leakage.

## 🎥 Demo

[▶️ Watch the Aster & Row AI Support Agent Demo](https://github.com/AlphaAustinCode/Aster-agent/releases/download/v1.0-demo/Aster-agent.Demo.mp4)

## ✨ Features

* 🧠 **Grounded RAG** — Answers are based on validated knowledge-base content.
* 🛡️ **Safe Indexing** — Blocks internal drafts and superseded policies.
* 📦 **Secure Order Lookup** — Exposes only customer-safe order information.
* 🧹 **Input Normalization** — Handles inputs such as `ord 1007` → `ORD-1007`.
* 🔐 **PII Protection** — Sensitive information is scrubbed from outputs and logs.
* 🚫 **LLM Guardrails** — Prevents unsupported claims and unsafe actions.
* 📚 **Smart Citations** — Requires citations when policy documents are used.
* 💬 **Session Memory** — Supports isolated multi-turn conversations.
* 📊 **Observability** — Tracks retrieval, turns, tools, and scrubbed outputs.
* 🧪 **Deterministic Evaluation** — Automated functional and security testing.
* 🤝 **Human Handoff** — Escalates when the agent cannot safely answer.

## 🏗️ Architecture

```text
                    Customer Query
                          │
                          ▼
                 ┌────────────────┐
                 │   Interface    │
                 └───────┬────────┘
                         ▼
                 ┌────────────────┐
                 │    Session     │
                 └───────┬────────┘
                         ▼
                 ┌────────────────┐
                 │   Safe RAG     │
                 │   Retrieval    │
                 └───────┬────────┘
                         │
               ┌─────────┴─────────┐
               ▼                   ▼
        Knowledge Base       Order Tool
               │                   │
               └─────────┬─────────┘
                         ▼
                 ┌────────────────┐
                 │   Guardrails   │
                 │  + Gemini LLM  │
                 └───────┬────────┘
                         ▼
                 ┌────────────────┐
                 │  Secure Output │
                 │ Answer/Source/ │
                 │    Handoff     │
                 └────────────────┘
```

### Core Layers

| Layer            | Component                   | Responsibility               |
| ---------------- | --------------------------- | ---------------------------- |
| 📚 RAG           | `src/rag/indexer.py`        | Safe indexing & embeddings   |
| 📦 Tools         | `src/tools/order_lookup.py` | Secure order information     |
| 🤖 Generation    | `src/generation/`           | Gemini + guardrails          |
| 💬 Sessions      | `session.py`                | Isolated conversation memory |
| 📊 Observability | `src/observability/`        | Structured telemetry         |
| 🧪 Evaluation    | `evaluation/`               | Automated validation         |

## 🔐 Security Design

The system follows a **defense-in-depth** approach:

```text
Knowledge Base
      ↓
Metadata Validation
      ↓
Safe Index
      ↓
Retrieval
      ↓
Tool / Policy Validation
      ↓
LLM Guardrails
      ↓
Citation Validation
      ↓
Customer-Safe Response
```

### Safe RAG

The indexer:

* Validates document metadata
* Splits Markdown into searchable chunks
* Generates embeddings
* Preserves provenance
* Excludes `superseded` policies
* Blocks internal `draft` content

### Secure Order Tool

Order IDs are normalized:

```text
ord 1007
    ↓
ORD-1007
```

Cancelled, returned, and failed orders automatically remove stale:

```text
estimated_delivery
delivery_date
carrier
tracking_number
```

## 📖 Smart Citation Guardrail

Citations are required when knowledge-base policy content is retrieved:

```text
[source: file.md#Heading]
```

Tool-only responses are **not** incorrectly forced to contain documentation citations.

This prevents false positives while maintaining grounded policy responses.

## 📁 Project Structure

```text
aster-row-agent/
├── knowledge-base/
├── src/
│   ├── rag/
│   │   └── indexer.py
│   ├── tools/
│   │   └── order_lookup.py
│   ├── generation/
│   │   ├── generator.py
│   │   ├── guardrails.py
│   │   └── session.py
│   ├── observability/
│   │   └── logger.py
│   └── interface/
│       └── cli.py
├── tests/
├── evaluation/
│   └── run_evaluation.py
├── requirements.txt
└── README.md
```

## ⚙️ Setup

### 1. Clone & Install

```bash
git clone <repository-url>
cd aster-row-agent

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure Gemini

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

> ⚠️ Never commit API keys or secrets to Git.

## ▶️ Run

```bash
python -m src.interface.cli
```

The agent returns:

```json
{
  "answer": "...",
  "sources": [],
  "human_handoff": false
}
```

## 🧪 Testing

Run the complete test suite:

```bash
pytest -q
```

Run the evaluation:

```bash
python evaluation/run_evaluation.py
```

The tests cover:

* Order ID normalization
* Secure order lookup
* Cancelled-order sanitization
* Citation validation
* Generation guardrails
* Session isolation
* Security refusals
* Tool interactions
* Response contract

## 🐛 Bug Fixes & Regression Tests

### Bug 1 — Stale Delivery Dates

**Problem:** Cancelled orders could expose old delivery dates.

**Fix:** Removed delivery-related fields for `cancelled`, `returned`, and `failed` orders.

**Regression Test:**

```text
tests/test_order_lookup.py::test_cancelled_order_strips_stale_delivery_fields
```

### Bug 2 — Citation False Positives

**Problem:** Tool-only responses were incorrectly required to contain policy citations.

**Fix:** Implemented **Smart Citation** logic.

**Regression Test:**

```text
tests/test_generation.py
```

### Bug 3 — Messy Order IDs

**Problem:** Inputs such as `ord 1007` failed lookup.

**Fix:** Added `sanitize_and_validate_order_id()` for normalization and validation.

**Regression Test:**

```text
tests/test_order_lookup.py::test_valid_order_id_normalization
```

## 🎯 Design Philosophy

> **Don't just make the AI answer — make sure it is safe, grounded, and allowed to answer.**

**Aster & Row AI Support Agent** — *Secure by design. Grounded by evidence. Reliable by evaluation.*
