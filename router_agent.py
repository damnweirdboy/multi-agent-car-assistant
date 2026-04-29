"""
agents/router_agent.py — Route Classification Agent

Classifies a user query into one of five categories:
  - buy          : user wants to buy or get a car recommendation
  - compare      : user wants to compare two or more cars
  - diagnose     : user is describing a car problem or symptom
  - maintenance  : user asks about service, upkeep, or intervals
  - general      : anything else

Strategy: keyword / phrase matching with confidence scoring.
No LLM cost — deterministic and fast.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from config import ROUTES


# ──────────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────────

@dataclass
class RouteResult:
    """Holds the classification output of the Router Agent."""
    category: str
    confidence: float               # 0.0 – 1.0
    matched_keywords: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        kw = ", ".join(self.matched_keywords) if self.matched_keywords else "none"
        return (
            f"category='{self.category}'  "
            f"confidence={self.confidence:.0%}  "
            f"keywords=[{kw}]"
        )


# ──────────────────────────────────────────────────────────────
# Keyword map  (ordered: more specific patterns first)
# ──────────────────────────────────────────────────────────────

_KEYWORD_MAP: dict[str, list[str]] = {
    "compare": [
        "compare", "vs", "versus", "difference between",
        "which is better", "which one", "corolla vs", "civic vs",
        "better than", "or the",
    ],
    "diagnose": [
        "overheating", "overheat", "engine light", "check engine",
        "knocking", "ticking", "noise", "vibration", "smoke",
        "not starting", "won't start", "wont start", "stalling",
        "oil pressure", "brake problem", "brake warning",
        "transmission slip", "gear slip", "rough idle",
        "squealing", "grinding", "warning light", "dashboard light",
        "something wrong", "problem", "issue", "broken", "leak",
        "battery dead", "battery warning",
    ],
    "maintenance": [
        "oil change", "change oil", "when to change", "service",
        "maintenance", "tune up", "tune-up", "tire rotation",
        "air filter", "spark plug", "brake pad", "coolant flush",
        "transmission fluid", "wiper blade", "timing belt",
        "how often", "interval", "schedule", "replace",
    ],
    "buy": [
        "buy", "purchase", "want a car", "recommend", "recommendation",
        "economical", "cheap", "affordable", "best car", "which car",
        "good car", "suggest", "looking for a car", "need a car",
        "first car", "family car", "budget car", "fuel efficient car",
        "low cost", "inexpensive",
    ],
}


# ──────────────────────────────────────────────────────────────
# Agent class
# ──────────────────────────────────────────────────────────────

class RouterAgent:
    """
    Agent 1 — Query Router.

    Classifies the user's question into a route category
    using keyword matching with simple confidence scoring.
    """

    def __init__(self) -> None:
        # Pre-compile all patterns for speed
        self._patterns: dict[str, list[re.Pattern]] = {
            category: [
                re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
                for kw in keywords
            ]
            for category, keywords in _KEYWORD_MAP.items()
        }

    # ----------------------------------------------------------
    def route(self, query: str) -> RouteResult:
        """
        Classify *query* and return a RouteResult.

        Parameters
        ----------
        query : str
            Raw user question.

        Returns
        -------
        RouteResult
            category, confidence, and matched keywords.
        """
        query = query.strip()
        scores: dict[str, list[str]] = {cat: [] for cat in _KEYWORD_MAP}

        # Count keyword hits per category
        for category, patterns in self._patterns.items():
            for i, pattern in enumerate(patterns):
                if pattern.search(query):
                    keyword = _KEYWORD_MAP[category][i]
                    scores[category].append(keyword)

        # Find the category with the most hits
        best_cat = max(scores, key=lambda c: len(scores[c]))
        best_hits = scores[best_cat]

        if not best_hits:
            # No keywords matched → general fallback
            return RouteResult(
                category="general",
                confidence=0.5,
                matched_keywords=[],
            )

        # Confidence: based on number of hits (caps at 1.0)
        # 1 hit → 0.70, 2 hits → 0.85, 3+ hits → 0.95
        hit_count = len(best_hits)
        confidence = min(0.60 + hit_count * 0.15, 0.98)

        return RouteResult(
            category=best_cat,
            confidence=confidence,
            matched_keywords=best_hits,
        )
