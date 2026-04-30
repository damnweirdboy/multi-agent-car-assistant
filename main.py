"""
main.py — Multi-Agent Car Assistant
====================================
Entry point for the pipeline.

Pipeline:
    User Question
        → Router Agent    (classifies query category)
        → Retriever Agent (RAG: finds relevant knowledge chunks)
        → Advisor Agent   (generates grounded answer)
        → Printed output

Usage:
    python main.py            # Interactive mode
    python main.py --demo     # Run built-in demo questions (no input needed)
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from typing import Optional

from config import USE_OPENAI
from agents import RouterAgent, RetrieverAgent, AdvisorAgent
from agents.retriever_agent import RetrievalResult
from agents.router_agent import RouteResult
from agents.advisor_agent import AdviceResult


# ══════════════════════════════════════════════════════════════
# Pretty-print helpers
# ══════════════════════════════════════════════════════════════

WIDTH = 70   # Console width

BANNER = f"""
{'═' * WIDTH}
  🚗  MULTI-AGENT CAR ASSISTANT
{'═' * WIDTH}
  Agents : Router  →  Retriever (RAG)  →  Advisor
  Mode   : {"OpenAI (" + __import__('config').OPENAI_MODEL + ")" if USE_OPENAI else "Fallback (no LLM — set OPENAI_API_KEY for AI answers)"}
{'═' * WIDTH}
"""

ROUTE_EMOJI = {
    "buy":         "💰 buy",
    "compare":     "⚖️  compare",
    "diagnose":    "🔧 diagnose",
    "maintenance": "🛠️  maintenance",
    "general":     "💬 general",
}


def _divider(char: str = "─", width: int = WIDTH) -> str:
    return char * width


def _print_route(route: RouteResult) -> None:
    label = ROUTE_EMOJI.get(route.category, route.category)
    kw = ", ".join(route.matched_keywords) if route.matched_keywords else "—"
    print(f"\n┌{'─' * (WIDTH - 2)}┐")
    print(f"│  STEP 1 — ROUTER AGENT{' ' * (WIDTH - 25)}│")
    print(f"├{'─' * (WIDTH - 2)}┤")
    print(f"│  Category  : {label:<{WIDTH - 17}}│")
    print(f"│  Confidence: {route.confidence:.0%}{' ' * (WIDTH - 17)}│")
    print(f"│  Keywords  : {kw:<{WIDTH - 17}}│")
    print(f"└{'─' * (WIDTH - 2)}┘")


def _print_retrieval(retrieval: RetrievalResult) -> None:
    print(f"\n┌{'─' * (WIDTH - 2)}┐")
    print(f"│  STEP 2 — RETRIEVER AGENT (RAG){' ' * (WIDTH - 34)}│")
    print(f"├{'─' * (WIDTH - 2)}┤")
    print(f"│  Found {len(retrieval.chunks)} chunk(s) for category "
          f"'{retrieval.route_category}'{' ' * max(0, WIDTH - 38 - len(retrieval.route_category))}│")
    print(f"├{'─' * (WIDTH - 2)}┤")

    if retrieval.chunks:
        for i, chunk in enumerate(retrieval.chunks, 1):
            preview = chunk.short_preview(max_chars=WIDTH - 20)
            src_line = f"  [{i}] [{chunk.source}] (sim={chunk.score:.3f})"
            print(f"│  {src_line:<{WIDTH - 4}}│")
            # Word-wrap the preview
            wrapped = textwrap.wrap(preview, width=WIDTH - 8)
            for line in wrapped:
                print(f"│      {line:<{WIDTH - 8}}│")
            if i < len(retrieval.chunks):
                print(f"│{' ' * (WIDTH - 2)}│")
    else:
        print(f"│  No relevant chunks found.{' ' * (WIDTH - 29)}│")

    print(f"└{'─' * (WIDTH - 2)}┘")


def _print_advice(advice: AdviceResult) -> None:
    print(f"\n┌{'─' * (WIDTH - 2)}┐")
    print(f"│  STEP 3 — ADVISOR AGENT  [mode: {advice.mode}]{' ' * max(0, WIDTH - 37 - len(advice.mode))}│")
    print(f"├{'─' * (WIDTH - 2)}┤")

    wrapped = textwrap.wrap(advice.answer, width=WIDTH - 6)
    for line in wrapped:
        print(f"│   {line:<{WIDTH - 5}}│")

    print(f"└{'─' * (WIDTH - 2)}┘\n")


def _run_query(
    question: str,
    router: RouterAgent,
    retriever: RetrieverAgent,
    advisor: AdvisorAgent,
) -> None:
    """Execute the full pipeline for a single question."""
    print(f"\n{'═' * WIDTH}")
    q_display = textwrap.shorten(question, width=WIDTH - 16, placeholder="…")
    print(f"  ❓ QUESTION: {q_display}")
    print(f"{'═' * WIDTH}")

    # Step 1: Route
    route: RouteResult = router.route(question)
    _print_route(route)

    # Step 2: Retrieve
    retrieval: RetrievalResult = retriever.retrieve(
        query=question,
        route_category=route.category,
    )
    _print_retrieval(retrieval)

    # Step 3: Advise
    advice: AdviceResult = advisor.advise(retrieval)
    _print_advice(advice)


# ══════════════════════════════════════════════════════════════
# Demo questions (covers all 5 routes)
# ══════════════════════════════════════════════════════════════

DEMO_QUESTIONS = [
    "I want an economical car for daily commuting",        # buy
    "Compare the Toyota Corolla and the Honda Civic",      # compare
    "My engine is overheating, what should I do?",         # diagnose
    "When should I change my engine oil?",                 # maintenance
    "Tell me something about cars",                        # general
]


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Agent Car Assistant — RAG + Multi-Agent Pipeline"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the 5 built-in demo questions instead of interactive mode.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild of the ChromaDB vector index.",
    )
    args = parser.parse_args()

    # ── Print banner ──────────────────────────────────────────
    print(BANNER)

    # ── Initialise agents ────────────────────────────────────
    print("[INIT] Initialising agents …")
    router = RouterAgent()
    retriever = RetrieverAgent()
    advisor = AdvisorAgent()

    # ── Build / load ChromaDB index ──────────────────────────
    print("\n[INIT] Building/loading knowledge index …")
    retriever.build_index(force_rebuild=args.rebuild)
    print("[INIT] ✓ System ready.\n")

    # ── Run questions ─────────────────────────────────────────
    if args.demo:
        print(f"Running {len(DEMO_QUESTIONS)} demo questions …")
        for question in DEMO_QUESTIONS:
            _run_query(question, router, retriever, advisor)
        print(f"{'═' * WIDTH}")
        print("  ✅ Demo complete.")
        print(f"{'═' * WIDTH}\n")
    else:
        # Interactive loop
        print("Type your car-related question and press Enter.")
        print('Type "exit" or press Ctrl+C to quit.\n')
        try:
            while True:
                print(_divider("─"))
                question = input("  Your question: ").strip()
                if not question:
                    continue
                if question.lower() in {"exit", "quit", "q"}:
                    print("\nGoodbye! 🚗\n")
                    break
                _run_query(question, router, retriever, advisor)
        except KeyboardInterrupt:
            print("\n\nGoodbye! 🚗\n")
            sys.exit(0)


if __name__ == "__main__":
    main()
