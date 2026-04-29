"""
config.py — Central configuration for Multi-Agent Car Assistant.
Edit this file to change models, paths, or API settings.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# ──────────────────────────────────────────────
# Embedding model (sentence-transformers, free / local)
# ──────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ──────────────────────────────────────────────
# ChromaDB collection name
# ──────────────────────────────────────────────
CHROMA_COLLECTION = "car_knowledge"

# ──────────────────────────────────────────────
# Retrieval settings
# ──────────────────────────────────────────────
TOP_K_RESULTS = 3          # Number of chunks to retrieve per query
CHUNK_SIZE = 200           # Approx. max words per chunk

# ──────────────────────────────────────────────
# LLM / OpenAI settings
# ──────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 512
TEMPERATURE = 0.3

# ──────────────────────────────────────────────
# Mode detection
# ──────────────────────────────────────────────
USE_OPENAI = bool(OPENAI_API_KEY)   # Automatically falls back if no key

# ──────────────────────────────────────────────
# Supported route categories
# ──────────────────────────────────────────────
ROUTES = ["buy", "compare", "diagnose", "maintenance", "general"]
