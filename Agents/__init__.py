"""
agents/__init__.py — Makes `agents` a Python package.
Exports the three core agents for easy import.
"""

from .router_agent import RouterAgent
from .retriever_agent import RetrieverAgent
from .advisor_agent import AdvisorAgent

__all__ = ["RouterAgent", "RetrieverAgent", "AdvisorAgent"]
