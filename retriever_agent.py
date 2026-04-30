"""
agents/retriever_agent.py — RAG Retriever Agent

Responsible for:
  1. Loading plain-text knowledge files from data/
  2. Splitting documents into overlapping chunks
  3. Embedding chunks with sentence-transformers
  4. Storing/loading a persistent ChromaDB vector collection
  5. Retrieving the top-K most relevant chunks for a user query

ChromaDB persists to disk (chroma_db/), so embedding only happens
once — subsequent runs load the existing index instantly.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_COLLECTION,
    CHROMA_DB_DIR,
    CHUNK_SIZE,
    DATA_DIR,
    EMBEDDING_MODEL,
    TOP_K_RESULTS,
)


# ──────────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A single retrieved passage from the knowledge base."""
    text: str
    source: str          # filename (e.g. "cars.txt")
    category: str        # route category label
    score: float         # cosine similarity distance (lower = better)
    chunk_id: str        # unique chunk identifier

    def short_preview(self, max_chars: int = 120) -> str:
        """Return a truncated preview of the text."""
        preview = self.text.replace("\n", " ").strip()
        return preview[:max_chars] + "…" if len(preview) > max_chars else preview


@dataclass
class RetrievalResult:
    """Holds all retrieved chunks for a query."""
    query: str
    route_category: str
    chunks: List[RetrievedChunk] = field(default_factory=list)

    @property
    def context_text(self) -> str:
        """Concatenate all chunk texts into a single context string."""
        parts = []
        for i, chunk in enumerate(self.chunks, 1):
            parts.append(f"[Source {i}: {chunk.source}]\n{chunk.text.strip()}")
        return "\n\n".join(parts)

    def __len__(self) -> int:
        return len(self.chunks)


# ──────────────────────────────────────────────────────────────
# Agent class
# ──────────────────────────────────────────────────────────────

class RetrieverAgent:
    """
    Agent 2 — RAG Retriever.

    Builds and queries a ChromaDB vector store containing
    car-domain knowledge from plain-text files.
    """

    def __init__(self) -> None:
        print("  [RetrieverAgent] Loading embedding model: "
              f"'{EMBEDDING_MODEL}' …")
        self._embedder = SentenceTransformer(EMBEDDING_MODEL)

        # Persistent ChromaDB client
        CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(CHROMA_DB_DIR),
        )

        self._collection: Optional[chromadb.Collection] = None

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def build_index(self, force_rebuild: bool = False) -> None:
        """
        Load data files, chunk them, embed, and store in ChromaDB.

        If the collection already exists and force_rebuild=False,
        this is a no-op (index is already on disk).
        """
        existing = [c.name for c in self._client.list_collections()]

        if CHROMA_COLLECTION in existing and not force_rebuild:
            print(f"  [RetrieverAgent] Index '{CHROMA_COLLECTION}' already "
                  "exists — skipping rebuild.")
            self._collection = self._client.get_collection(CHROMA_COLLECTION)
            return

        # Delete stale collection if rebuilding
        if CHROMA_COLLECTION in existing:
            self._client.delete_collection(CHROMA_COLLECTION)

        self._collection = self._client.create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

        all_chunks = self._load_and_chunk_documents()
        if not all_chunks:
            raise RuntimeError(
                f"No documents found in '{DATA_DIR}'. "
                "Make sure the data/ directory contains .txt files."
            )

        texts = [c["text"] for c in all_chunks]
        ids = [c["id"] for c in all_chunks]
        metadatas = [c["metadata"] for c in all_chunks]

        print(f"  [RetrieverAgent] Embedding {len(texts)} chunks …")
        embeddings = self._embedder.encode(texts, show_progress_bar=True).tolist()

        self._collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        print(f"  [RetrieverAgent] ✓ Indexed {len(texts)} chunks into "
              f"'{CHROMA_COLLECTION}'.")

    def retrieve(
        self,
        query: str,
        route_category: str,
        top_k: int = TOP_K_RESULTS,
    ) -> RetrievalResult:
        """
        Retrieve the top-K most relevant chunks for *query*.

        Parameters
        ----------
        query          : User's question.
        route_category : Category from RouterAgent (used for metadata filter).
        top_k          : Number of chunks to return.

        Returns
        -------
        RetrievalResult containing matched chunks and context text.
        """
        if self._collection is None:
            raise RuntimeError(
                "Index not built. Call build_index() before retrieve()."
            )

        query_embedding = self._embedder.encode([query]).tolist()

        # If route is 'general', search all docs; otherwise filter by category
        where_filter = (
            {"category": route_category}
            if route_category != "general"
            else None
        )

        query_kwargs = dict(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        if where_filter:
            query_kwargs["where"] = where_filter

        results = self._collection.query(**query_kwargs)

        chunks: List[RetrievedChunk] = []
        for doc, meta, dist, cid in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0],
        ):
            chunks.append(
                RetrievedChunk(
                    text=doc,
                    source=meta.get("source", "unknown"),
                    category=meta.get("category", "general"),
                    score=round(1.0 - dist, 4),   # Convert distance → similarity
                    chunk_id=cid,
                )
            )

        return RetrievalResult(
            query=query,
            route_category=route_category,
            chunks=chunks,
        )

    # ----------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------

    def _load_and_chunk_documents(self) -> list[dict]:
        """
        Read all .txt files in DATA_DIR and split into chunks.

        Each paragraph (blank-line-separated block) is treated as a
        natural chunk unit. Long paragraphs are further split by
        sentence boundary to stay under CHUNK_SIZE words.

        Returns a list of dicts with keys: id, text, metadata.
        """
        data_files = list(DATA_DIR.glob("*.txt"))
        if not data_files:
            return []

        all_chunks: list[dict] = []
        chunk_idx = 0

        for filepath in sorted(data_files):
            source_name = filepath.name
            raw_text = filepath.read_text(encoding="utf-8")

            # Each paragraph is separated by a blank line
            paragraphs = re.split(r"\n\s*\n", raw_text.strip())

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                # Extract optional CATEGORY: tag from the paragraph header
                category = self._extract_category(para)
                # Strip the CATEGORY line from the text body
                body = re.sub(r"^CATEGORY:\s*\w+\s*\n?", "", para, flags=re.IGNORECASE).strip()

                # Split long paragraphs into sentence-bounded sub-chunks
                sub_chunks = self._split_into_chunks(body, CHUNK_SIZE)

                for sub in sub_chunks:
                    if not sub.strip():
                        continue
                    all_chunks.append({
                        "id": f"chunk_{chunk_idx:04d}",
                        "text": sub.strip(),
                        "metadata": {
                            "source": source_name,
                            "category": category,
                        },
                    })
                    chunk_idx += 1

        return all_chunks

    @staticmethod
    def _extract_category(paragraph: str) -> str:
        """Pull the CATEGORY tag from a paragraph, defaulting to 'general'."""
        match = re.match(r"CATEGORY:\s*(\w+)", paragraph, re.IGNORECASE)
        if match:
            cat = match.group(1).lower()
            from config import ROUTES
            return cat if cat in ROUTES else "general"
        return "general"

    @staticmethod
    def _split_into_chunks(text: str, max_words: int) -> list[str]:
        """
        Split text into chunks of at most max_words words,
        breaking on sentence boundaries where possible.
        """
        # Simple sentence splitter: split on '. ', '! ', '? '
        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks: list[str] = []
        current: list[str] = []
        current_words = 0

        for sentence in sentences:
            word_count = len(sentence.split())
            if current_words + word_count > max_words and current:
                chunks.append(" ".join(current))
                current = []
                current_words = 0
            current.append(sentence)
            current_words += word_count

        if current:
            chunks.append(" ".join(current))

        return chunks
