# 🚗 Multi-Agent Car Assistant

A university assignment project demonstrating **Multi-Agent Systems** and **Retrieval-Augmented Generation (RAG)** applied to the car domain.

---

## Project Overview

The system answers car-related questions through a three-agent pipeline:

```
User Question
    ↓
┌─────────────────┐
│  Router Agent   │  ← Classifies query: buy / compare / diagnose / maintenance / general
└────────┬────────┘
         ↓
┌─────────────────┐
│ Retriever Agent │  ← RAG: searches ChromaDB for relevant knowledge chunks
│     (RAG)       │
└────────┬────────┘
         ↓
┌─────────────────┐
│  Advisor Agent  │  ← Generates a grounded answer (OpenAI or fallback mode)
└────────┬────────┘
         ↓
┌─────────────────┐
│  Monitor Agent  │  ← Audits the answer for safety and groundedness (hallucinations)
└────────┬────────┘
         ↓
    Final Answer
```

---

## Agents

### 1. Router Agent (`agents/router_agent.py`)
- Classifies the user query using **keyword matching + confidence scoring**
- Returns: `category`, `confidence`, `matched_keywords`
- Categories: `buy`, `compare`, `diagnose`, `maintenance`, `general`

### 2. Retriever Agent — RAG (`agents/retriever_agent.py`)
- Loads `.txt` knowledge files from `data/`
- **Chunks** documents by paragraph + sentence boundaries
- **Embeds** chunks using `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Stores** embeddings in a persistent **ChromaDB** vector database
- **Retrieves** top-K most relevant chunks via cosine similarity
- Filters by route category metadata for better precision

### 3. Advisor Agent (`agents/advisor_agent.py`)
- **OpenAI mode**: Calls `gpt-3.5-turbo` with a strict system prompt that constrains answers to the retrieved context only
- **Fallback mode**: Formats retrieved chunks into a structured answer (no API key required)
- Responds with `"Not enough information"` when context is insufficient

### 4. Monitor Agent (`agents/monitor_agent.py`)
- **Safety Check**: Locally audits the generated text for dangerous content and restricted keywords.
- **Groundedness Check**: Evaluates lexical overlap between the generated answer and the source context to detect hallucinations.
- Operates entirely locally for maximum performance and security.

---

## Project Structure

```
multi-agent-car-assistant/
├── data/
│   ├── cars.txt         ← Car recommendations & comparisons
│   ├── diagnosis.txt    ← Engine & mechanical fault diagnosis
│   └── maintenance.txt  ← Service intervals & maintenance advice
├── agents/
│   ├── __init__.py
│   ├── router_agent.py
│   ├── retriever_agent.py
│   └── advisor_agent.py
├── chroma_db/           ← Auto-created: persistent vector store
├── main.py              ← Pipeline orchestrator (entry point)
├── config.py            ← Configuration (paths, models, API settings)
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate          # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~80 MB) on first run. After that it is cached locally.

### 3. (Optional) Set OpenAI API key
```bash
export OPENAI_API_KEY="sk-..."   # macOS / Linux
set OPENAI_API_KEY=sk-...        # Windows
```
If not set, the system runs in **fallback mode** — the full agent pipeline still works; the Advisor Agent returns formatted retrieved chunks instead of LLM-generated text.

---

## Usage

### Interactive mode
```bash
python main.py
```

### Demo mode (5 built-in questions, no input needed)
```bash
python main.py --demo
```

### Force rebuild of the ChromaDB index
```bash
python main.py --rebuild
```

---

## Example Output

```
══════════════════════════════════════════════════════════════════════
  ❓ QUESTION: My engine is overheating, what should I do?
══════════════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────────┐
│  STEP 1 — ROUTER AGENT                                           │
├──────────────────────────────────────────────────────────────────┤
│  Category  : 🔧 diagnose                                         │
│  Confidence: 85%                                                 │
│  Keywords  : overheating, engine                                 │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  STEP 2 — RETRIEVER AGENT (RAG)                                  │
├──────────────────────────────────────────────────────────────────┤
│  Found 3 chunk(s) for category 'diagnose'                        │
├──────────────────────────────────────────────────────────────────┤
│  [1] [diagnosis.txt] (sim=0.872)                                 │
│      Engine overheating is a serious problem that must be…       │
│  [2] [diagnosis.txt] (sim=0.841)                                 │
│      Low coolant / antifreeze is a common cause of engine…       │
│  [3] [diagnosis.txt] (sim=0.798)                                 │
│      Radiator problems can cause overheating and cooling…        │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  STEP 3 — ADVISOR AGENT  [mode: openai]                          │
├──────────────────────────────────────────────────────────────────┤
│   If your engine is overheating, stop safely and turn off        │
│   the engine. Do not open the radiator cap while hot. Common     │
│   causes include low coolant, a faulty thermostat, a blocked     │
│   radiator, or a broken water pump. Check the coolant level      │
│   first once the engine cools.                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Example Questions by Category

| Category | Example Questions |
|---|---|
| `buy` | "I want an economical car", "Recommend a cheap first car" |
| `compare` | "Compare Corolla and Civic", "Toyota vs BMW" |
| `diagnose` | "My engine is overheating", "There's a knocking noise" |
| `maintenance` | "When should I change my oil?", "How often to rotate tires?" |
| `general` | "Tell me about cars" |

---

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector DB | ChromaDB (persistent, file-based) |
| LLM (optional) | OpenAI `gpt-3.5-turbo` |
| Routing | Keyword-based classifier |

---

## Notes for the Assignment

- **Router Agent**: Implements keyword-based intent classification with confidence scoring. No LLM cost. Deterministic output for reproducible demos.
- **Retriever Agent (RAG)**: Full RAG implementation — load → chunk → embed → store → retrieve. ChromaDB persists the index to disk, so re-runs are instant.
- **Advisor Agent**: Demonstrates context-grounded generation with OpenAI, and a no-LLM fallback for environments without API access.
- **Robust Knowledge Base**: The `data/` folder currently contains over 90 highly detailed, structured entries covering car recommendations, mechanical diagnostics, and maintenance schedules. The system will automatically chunk and index any new `.txt` files added here on the next `--rebuild` run.
