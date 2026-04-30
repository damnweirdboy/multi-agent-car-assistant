"""
agents/advisor_agent.py — Answer Generation Agent

Takes the retrieved context from RetrieverAgent and generates
a grounded, helpful answer for the user.

Two modes (auto-detected via config.USE_OPENAI):
  1. OpenAI mode  — calls gpt-3.5-turbo with a strict system prompt
                    that constrains answers to the retrieved context.
  2. Fallback mode — formats and returns the retrieved chunks directly.
                    No LLM cost; still demonstrates the full pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import (
    MAX_TOKENS,
    OPENAI_MODEL,
    TEMPERATURE,
    USE_OPENAI,
)
from agents.retriever_agent import RetrievalResult


# ──────────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────────

@dataclass
class AdviceResult:
    """Final answer produced by the Advisor Agent."""
    answer: str
    mode: str           # "openai" | "fallback"
    used_context: bool  # False when no chunks were retrieved

    def __str__(self) -> str:
        return self.answer


# ──────────────────────────────────────────────────────────────
# System prompt for OpenAI mode
# ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert car assistant. Your job is to answer the user's question
using ONLY the information provided in the context below.

Rules:
- Be concise, clear, and helpful.
- If the context does not contain enough information to answer the question,
  respond with exactly: "Not enough information in the knowledge base to answer this question."
- Do NOT make up information not present in the context.
- Format your answer in plain text (no markdown).
- Keep answers under 200 words.
"""


# ──────────────────────────────────────────────────────────────
# Agent class
# ──────────────────────────────────────────────────────────────

class AdvisorAgent:
    """
    Agent 3 — Advisor / Answer Generator.

    Produces a final answer grounded in the retrieved context.
    Automatically uses OpenAI if OPENAI_API_KEY is set,
    otherwise falls back to structured chunk formatting.
    """

    def __init__(self) -> None:
        self._openai_client = None

        if USE_OPENAI:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI()
                print(f"  [AdvisorAgent] Mode: OpenAI ({OPENAI_MODEL})")
            except ImportError:
                print("  [AdvisorAgent] openai package not installed — "
                      "falling back to no-LLM mode.")
        else:
            print("  [AdvisorAgent] Mode: Fallback (no OpenAI key detected)")

    # ----------------------------------------------------------
    def advise(self, retrieval: RetrievalResult) -> AdviceResult:
        """
        Generate an answer from the retrieval result.

        Parameters
        ----------
        retrieval : RetrievalResult
            Output from RetrieverAgent.retrieve().

        Returns
        -------
        AdviceResult with the final answer text.
        """
        if not retrieval.chunks:
            return AdviceResult(
                answer="Not enough information in the knowledge base "
                       "to answer this question.",
                mode="no-context",
                used_context=False,
            )

        if self._openai_client:
            return self._answer_with_openai(retrieval)
        else:
            return self._answer_fallback(retrieval)

    # ----------------------------------------------------------
    # OpenAI mode
    # ----------------------------------------------------------

    def _answer_with_openai(self, retrieval: RetrievalResult) -> AdviceResult:
        """Call gpt-3.5-turbo with the retrieved context."""
        user_message = (
            f"Context:\n{retrieval.context_text}\n\n"
            f"Question: {retrieval.query}"
        )

        response = self._openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )

        answer = response.choices[0].message.content.strip()
        return AdviceResult(answer=answer, mode="openai", used_context=True)

    # ----------------------------------------------------------
    # Fallback mode (no LLM)
    # ----------------------------------------------------------

    def _answer_fallback(self, retrieval: RetrievalResult) -> AdviceResult:
        """
        Format the retrieved chunks into a structured answer.
        No LLM required — useful for demo / no-API-key scenarios.
        """
        lines = [
            f"Based on the car knowledge base ({retrieval.route_category} category):\n"
        ]

        for i, chunk in enumerate(retrieval.chunks, 1):
            # Take the first two sentences of each chunk as a summary
            sentences = chunk.text.replace("\n", " ").split(". ")
            summary = ". ".join(sentences[:2]).strip()
            if not summary.endswith("."):
                summary += "."
            lines.append(f"  {i}. {summary}")

        lines.append(
            f"\n[Tip: Set the OPENAI_API_KEY environment variable for "
            "more detailed AI-generated answers.]"
        )

        return AdviceResult(
            answer="\n".join(lines),
            mode="fallback",
            used_context=True,
        )
