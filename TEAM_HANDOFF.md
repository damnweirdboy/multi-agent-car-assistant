# Team Handoff — Multi-Agent Car Assistant

## What's in the ZIP

```
multi-agent-car-assistant/
├── data/                      ← Car knowledge base (plain text files)
│   ├── cars.txt               ← Recommendations & comparisons
│   ├── diagnosis.txt          ← Engine/fault diagnosis
│   └── maintenance.txt        ← Service intervals & upkeep
├── agents/
│   ├── router_agent.py        ← AGENT 1: Query classifier
│   ├── retriever_agent.py     ← AGENT 2: RAG (ChromaDB + embeddings)
│   └── advisor_agent.py       ← AGENT 3: Answer generation
├── main.py                    ← Pipeline entry point
├── config.py                  ← All settings (models, paths, keys)
└── requirements.txt           ← Python dependencies
```

---

## Quick Setup (everyone on the team)

```bash
# 1. Unzip the project
unzip multi-agent-car-assistant.zip
cd multi-agent-car-assistant

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the demo
python3 main.py --demo
```

---

## What Each Team Member Needs

### 🟢 Your Part (already done)
| Component | File | Status |
|---|---|---|
| Router Agent | `agents/router_agent.py` | ✅ Complete |
| Retriever Agent (RAG) | `agents/retriever_agent.py` | ✅ Complete |
| Advisor Agent | `agents/advisor_agent.py` | ✅ Complete |
| Pipeline | `main.py` | ✅ Complete |
| Knowledge Base | `data/*.txt` | ✅ Complete |

---

### 🔵 Monitoring Team — What to Hook Into

Your team members doing **monitoring** need to plug into the pipeline in `main.py` inside the `_run_query()` function.

**Key data to monitor** (all already available as variables):

```python
# After Router Agent runs:
route.category          # str: "buy" / "compare" / "diagnose" / "maintenance" / "general"
route.confidence        # float: 0.0 – 1.0
route.matched_keywords  # list[str]

# After Retriever Agent runs:
retrieval.chunks        # list of RetrievedChunk objects
retrieval.chunks[i].score    # cosine similarity score
retrieval.chunks[i].source   # which file it came from
retrieval.chunks[i].text     # the actual chunk text

# After Advisor Agent runs:
advice.answer           # final answer string
advice.mode             # "openai" or "fallback"
advice.used_context     # bool
```

**Suggested monitoring additions:**
- Log each query + route + answer to a `.jsonl` file
- Track response latency per agent step
- Flag low-confidence routes (< 0.6) for review
- Count how often "Not enough information" is returned

---

### 🔴 Evaluation Team — What to Hook Into

**Evaluation** should test these things:

1. **Router accuracy** — Does the right category get assigned?
   - File to test: `agents/router_agent.py` → `RouterAgent.route(query)`
   - Create a test set of 20–30 questions with known categories

2. **Retrieval quality** — Are the right chunks returned?
   - File to test: `agents/retriever_agent.py` → `RetrieverAgent.retrieve(query, category)`
   - Check: Is the top chunk actually relevant? Is sim score > 0.5?

3. **Answer quality** — Is the Advisor grounded?
   - File to test: `agents/advisor_agent.py` → `AdvisorAgent.advise(retrieval)`
   - Check: Does the answer contain info from the retrieved chunks?

**Minimal evaluation script example:**
```python
from agents import RouterAgent, RetrieverAgent, AdvisorAgent

router = RouterAgent()
retriever = RetrieverAgent()
retriever.build_index()
advisor = AdvisorAgent()

test_cases = [
    ("I want a cheap car", "buy"),
    ("Compare Corolla and Civic", "compare"),
    ("My engine is overheating", "diagnose"),
    ("When to change oil?", "maintenance"),
]

correct = 0
for question, expected in test_cases:
    result = router.route(question)
    if result.category == expected:
        correct += 1
    print(f"Q: {question}")
    print(f"  Expected: {expected} | Got: {result.category} ({'✅' if result.category == expected else '❌'})")

print(f"\nRouter Accuracy: {correct}/{len(test_cases)} = {correct/len(test_cases):.0%}")
```

---

## Adding an OpenAI Key (for better answers)

```bash
export OPENAI_API_KEY="sk-..."   # Mac/Linux
python3 main.py
```

Without a key the system runs in **fallback mode** — still fully functional for demo/evaluation purposes.

---

## Extending the Knowledge Base

To add more car knowledge:
1. Add or edit `.txt` files in `data/`
2. Start each paragraph with `CATEGORY: buy/compare/diagnose/maintenance`
3. Rebuild the index: `python3 main.py --rebuild`

---

## Contact

Built by [Emir — covers Agents + RAG.
Monitoring & Evaluation to be added by the team.
