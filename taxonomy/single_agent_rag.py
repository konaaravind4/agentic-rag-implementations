"""
taxonomy/single_agent_rag.py
============================
Single-Agent Agentic RAG: Router (§5.1)
----------------------------------------
Architecture from: "Agentic RAG: A Survey" (arXiv:2501.09136)

In Single-Agent Agentic RAG, a single centralized agent inspects each
incoming query, selects the most appropriate retrieval tool from its
toolkit, executes the retrieval, and synthesizes a final answer.  The
"routing" decision is made by the LLM itself via a structured prompt,
which reasons about query intent before committing to a tool.

Retrieval toolkit
-----------------
1. semantic_search   – dense FAISS vector similarity search
2. sql_search        – structured query over an in-memory SQLite DB
3. keyword_search    – sparse BM25-style term-frequency search
4. recommendation    – content-based recommendation (cosine similarity)
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import time
from collections import Counter
from typing import Any

# ---------------------------------------------------------------------------
# Project-level imports (assumed to be available in the wider project)
# ---------------------------------------------------------------------------
from core.embeddings import Embedder          # noqa: F401  (type hint only)
from core.vector_store import FAISSVectorStore
from core.llm import LocalLLM

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Corpus used by keyword and recommendation tools (stand-alone fallback)
# ---------------------------------------------------------------------------
_DEMO_CORPUS: list[dict] = [
    {"id": 1, "text": "Machine learning models require large amounts of training data.",
     "topic": "ml"},
    {"id": 2, "text": "Neural networks are inspired by the human brain structure.",
     "topic": "ml"},
    {"id": 3, "text": "The stock market showed significant gains this quarter.",
     "topic": "finance"},
    {"id": 4, "text": "Interest rates affect mortgage payments and home buying.",
     "topic": "finance"},
    {"id": 5, "text": "Climate change impacts global weather patterns dramatically.",
     "topic": "science"},
    {"id": 6, "text": "Renewable energy sources include solar and wind power.",
     "topic": "science"},
    {"id": 7, "text": "Healthy diet includes vegetables, fruits, and lean proteins.",
     "topic": "health"},
    {"id": 8, "text": "Regular exercise improves cardiovascular health significantly.",
     "topic": "health"},
]

# ---------------------------------------------------------------------------
# Tool-routing prompt template
# ---------------------------------------------------------------------------
_ROUTING_PROMPT = """You are a query router for a RAG (Retrieval-Augmented Generation) system.
Your task is to select the SINGLE best retrieval tool for the user's question.

Available tools:
  - semantic_search   : Best for conceptual, open-ended, or natural-language questions.
  - sql_search        : Best for structured queries about facts, numbers, or records.
  - keyword_search    : Best for exact keyword or terminology lookup.
  - recommendation    : Best when the user wants similar or related content suggestions.

User question: "{query}"

Respond with a JSON object ONLY in the format:
{{
  "tool": "<tool_name>",
  "reasoning": "<one sentence explaining why>"
}}
"""

# ---------------------------------------------------------------------------
# Helper: Simple BM25-style scorer
# ---------------------------------------------------------------------------

def _bm25_score(query: str, doc_text: str, k1: float = 1.5, b: float = 0.75,
                avg_dl: float = 10.0) -> float:
    """Return a BM25-inspired relevance score between a query and a document."""
    query_terms = query.lower().split()
    doc_terms = doc_text.lower().split()
    dl = len(doc_terms)
    tf_map = Counter(doc_terms)
    score = 0.0
    for term in query_terms:
        tf = tf_map.get(term, 0)
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * dl / avg_dl)
        score += numerator / (denominator + 1e-9)
    return score


# ---------------------------------------------------------------------------
# Helper: Cosine similarity on raw term vectors (no embeddings required)
# ---------------------------------------------------------------------------

def _term_vector(text: str) -> dict[str, float]:
    """Create a simple TF vector from text."""
    tokens = re.findall(r"\w+", text.lower())
    counts = Counter(tokens)
    total = sum(counts.values()) or 1
    return {t: c / total for t, c in counts.items()}


def _cosine(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Compute cosine similarity between two term-frequency vectors."""
    dot = sum(vec_a.get(t, 0.0) * vec_b.get(t, 0.0) for t in vec_b)
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values())) or 1e-9
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values())) or 1e-9
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SingleAgentRAG:
    """
    Single-Agent Agentic RAG: Router (§5.1).

    A single LLM-driven agent examines the incoming query, selects the
    most appropriate retrieval tool from its toolkit, retrieves context,
    and synthesizes a final answer.

    Parameters
    ----------
    llm : LocalLLM
        The language model used for routing decisions and answer synthesis.
    vector_store : FAISSVectorStore
        Pre-built FAISS index for semantic search.
    k : int
        Number of documents to retrieve per tool call (default 3).
    """

    # Names of all registered retrieval tools
    TOOL_NAMES = ("semantic_search", "sql_search", "keyword_search", "recommendation")

    def __init__(self, llm: LocalLLM, vector_store: FAISSVectorStore, k: int = 3) -> None:
        self.llm = llm
        self.vector_store = vector_store
        self.k = k

        # Build the in-memory SQLite demo database
        self._db_conn = self._init_sqlite()
        logger.info("SingleAgentRAG initialised (k=%d)", k)

    # ------------------------------------------------------------------
    # SQLite setup
    # ------------------------------------------------------------------

    @staticmethod
    def _init_sqlite() -> sqlite3.Connection:
        """
        Create an in-memory SQLite database populated with sample records
        spanning several domains (ML, finance, science, health).
        """
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id      INTEGER PRIMARY KEY,
                text    TEXT    NOT NULL,
                topic   TEXT    NOT NULL,
                score   REAL    DEFAULT 0.0
            )
            """
        )
        for doc in _DEMO_CORPUS:
            cursor.execute(
                "INSERT INTO documents (id, text, topic, score) VALUES (?, ?, ?, ?)",
                (doc["id"], doc["text"], doc["topic"], 0.0),
            )
        conn.commit()
        logger.debug("SQLite in-memory DB initialised with %d records.", len(_DEMO_CORPUS))
        return conn

    # ------------------------------------------------------------------
    # Tool 1: Semantic Search (FAISS)
    # ------------------------------------------------------------------

    def semantic_search(self, query: str) -> list[dict]:
        """
        Dense vector similarity search using the FAISS index.

        Parameters
        ----------
        query : str
            Natural-language search query.

        Returns
        -------
        list[dict]
            Up to ``self.k`` documents with keys ``text``, ``score``, ``source``.
        """
        logger.info("[Tool] semantic_search | query=%r", query)
        results = self.vector_store.search(query, top_k=self.k)
        docs = []
        for item in results:
            docs.append(
                {
                    "text": item.get("text", item.get("content", str(item))),
                    "score": float(item.get("score", item.get("similarity", 0.0))),
                    "source": "faiss_vector_store",
                }
            )
        return docs

    # ------------------------------------------------------------------
    # Tool 2: SQL Search (SQLite)
    # ------------------------------------------------------------------

    def sql_search(self, query: str) -> list[dict]:
        """
        Full-text keyword search inside the SQLite demo database using
        a LIKE predicate.  For a production system this would be replaced
        with proper SQL query generation via the LLM.

        Parameters
        ----------
        query : str
            The user question (keywords are extracted automatically).

        Returns
        -------
        list[dict]
            Matching rows as dicts with keys ``id``, ``text``, ``topic``, ``source``.
        """
        logger.info("[Tool] sql_search | query=%r", query)
        keywords = [w.strip() for w in re.findall(r"\w{3,}", query.lower())]
        if not keywords:
            keywords = [""]

        cursor = self._db_conn.cursor()
        results: list[dict] = []
        seen_ids: set[int] = set()

        for kw in keywords:
            cursor.execute(
                "SELECT id, text, topic FROM documents WHERE lower(text) LIKE ? LIMIT ?",
                (f"%{kw}%", self.k),
            )
            for row in cursor.fetchall():
                doc_id, text, topic = row
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    results.append(
                        {"id": doc_id, "text": text, "topic": topic, "source": "sqlite_db"}
                    )
            if len(results) >= self.k:
                break

        logger.debug("[sql_search] returned %d rows", len(results))
        return results[: self.k]

    # ------------------------------------------------------------------
    # Tool 3: Keyword Search (BM25-style)
    # ------------------------------------------------------------------

    def keyword_search(self, query: str) -> list[dict]:
        """
        Sparse BM25-style keyword retrieval over the in-memory demo corpus.

        Parameters
        ----------
        query : str
            Natural-language or keyword query.

        Returns
        -------
        list[dict]
            Top-k documents sorted by BM25 relevance score.
        """
        logger.info("[Tool] keyword_search | query=%r", query)
        avg_dl = sum(len(d["text"].split()) for d in _DEMO_CORPUS) / len(_DEMO_CORPUS)
        scored = [
            {
                "text": doc["text"],
                "topic": doc["topic"],
                "score": _bm25_score(query, doc["text"], avg_dl=avg_dl),
                "source": "bm25_keyword_search",
            }
            for doc in _DEMO_CORPUS
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[: self.k]

    # ------------------------------------------------------------------
    # Tool 4: Recommendation (content-based cosine similarity)
    # ------------------------------------------------------------------

    def recommendation(self, query: str) -> list[dict]:
        """
        Content-based recommendation: returns corpus documents most similar
        to the query using term-frequency cosine similarity.

        Parameters
        ----------
        query : str
            A description of the content the user wants recommendations for.

        Returns
        -------
        list[dict]
            Top-k recommended documents with cosine similarity scores.
        """
        logger.info("[Tool] recommendation | query=%r", query)
        query_vec = _term_vector(query)
        scored = []
        for doc in _DEMO_CORPUS:
            doc_vec = _term_vector(doc["text"])
            sim = _cosine(query_vec, doc_vec)
            scored.append(
                {
                    "text": doc["text"],
                    "topic": doc["topic"],
                    "score": sim,
                    "source": "content_based_recommendation",
                }
            )
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[: self.k]

    # ------------------------------------------------------------------
    # Routing logic
    # ------------------------------------------------------------------

    def _decide_tool(self, query: str) -> tuple[str, str]:
        """
        Ask the LLM to route the query to the most appropriate tool.

        The LLM is presented with a structured prompt listing all available
        tools and must respond with a JSON object specifying the chosen tool
        and its reasoning.  If parsing fails, a safe fallback
        (``semantic_search``) is used.

        Parameters
        ----------
        query : str
            The original user question.

        Returns
        -------
        tuple[str, str]
            ``(tool_name, reasoning)`` where ``tool_name`` is one of
            ``TOOL_NAMES`` and ``reasoning`` is the LLM's explanation.
        """
        prompt = _ROUTING_PROMPT.format(query=query)
        raw_response: str = self.llm.generate(prompt)

        # Attempt to parse the JSON decision
        try:
            # Extract JSON even if the LLM wraps it in extra text
            json_match = re.search(r"\{.*?\}", raw_response, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group())
            else:
                decision = json.loads(raw_response)

            tool_name = decision.get("tool", "semantic_search").strip().lower()
            reasoning = decision.get("reasoning", "No reasoning provided.")

            if tool_name not in self.TOOL_NAMES:
                logger.warning(
                    "LLM returned unknown tool %r; falling back to semantic_search", tool_name
                )
                tool_name = "semantic_search"
        except (json.JSONDecodeError, AttributeError) as exc:
            logger.warning("Could not parse routing JSON (%s); defaulting to semantic_search", exc)
            tool_name = "semantic_search"
            reasoning = "Fallback: JSON parse error."

        logger.info(
            "Routing decision | tool=%r | reasoning=%r", tool_name, reasoning
        )
        return tool_name, reasoning

    # ------------------------------------------------------------------
    # Execution dispatch
    # ------------------------------------------------------------------

    def _execute_retrieval(self, tool_name: str, query: str) -> list[dict]:
        """
        Dispatch execution to the selected retrieval tool.

        Parameters
        ----------
        tool_name : str
            One of ``TOOL_NAMES``.
        query : str
            The user query.

        Returns
        -------
        list[dict]
            Retrieved documents from the chosen tool.

        Raises
        ------
        ValueError
            If ``tool_name`` is not a registered tool.
        """
        dispatch: dict[str, Any] = {
            "semantic_search": self.semantic_search,
            "sql_search": self.sql_search,
            "keyword_search": self.keyword_search,
            "recommendation": self.recommendation,
        }
        if tool_name not in dispatch:
            raise ValueError(f"Unknown tool: {tool_name!r}. Must be one of {self.TOOL_NAMES}.")
        return dispatch[tool_name](query)

    # ------------------------------------------------------------------
    # Answer synthesis
    # ------------------------------------------------------------------

    def _synthesize_answer(self, query: str, docs: list[dict], tool_name: str) -> str:
        """
        Generate a final answer using the LLM and the retrieved context.

        Parameters
        ----------
        query : str
            Original user question.
        docs : list[dict]
            Retrieved documents from the chosen tool.
        tool_name : str
            Name of the tool that produced the docs.

        Returns
        -------
        str
            The LLM-generated answer.
        """
        context_blocks = []
        for i, doc in enumerate(docs, start=1):
            context_blocks.append(f"[Doc {i}] {doc.get('text', '')}")
        context_str = "\n".join(context_blocks) if context_blocks else "No context retrieved."

        synthesis_prompt = (
            f"You are a helpful assistant. Answer the question using ONLY the provided context.\n\n"
            f"Context (retrieved via {tool_name}):\n{context_str}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
        return self.llm.generate(synthesis_prompt)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def query(self, question: str) -> dict:
        """
        Route the question, retrieve relevant context, and synthesize an answer.

        This is the main entry point for the Single-Agent RAG Router.
        Internally it performs three steps:

        1. **Route** – the agent's LLM decides which tool best fits the query.
        2. **Retrieve** – the chosen tool fetches up to ``k`` documents.
        3. **Synthesize** – the LLM generates a grounded answer from the context.

        Parameters
        ----------
        question : str
            The user's natural-language question.

        Returns
        -------
        dict
            A result dictionary with the following keys:

            * ``tool_used``      – name of the retrieval tool selected.
            * ``reasoning``      – the agent's explanation for the tool choice.
            * ``retrieved_docs`` – list of retrieved document dicts.
            * ``answer``         – the synthesized answer string.
            * ``latency``        – end-to-end wall-clock latency in seconds.
        """
        logger.info("=== SingleAgentRAG.query | question=%r ===", question)
        t_start = time.perf_counter()

        # Step 1: Route
        tool_name, reasoning = self._decide_tool(question)

        # Step 2: Retrieve
        retrieved_docs = self._execute_retrieval(tool_name, question)
        logger.info("Retrieved %d documents via %r", len(retrieved_docs), tool_name)

        # Step 3: Synthesize
        answer = self._synthesize_answer(question, retrieved_docs, tool_name)

        latency = time.perf_counter() - t_start
        logger.info("Query completed in %.3f s", latency)

        return {
            "tool_used": tool_name,
            "reasoning": reasoning,
            "retrieved_docs": retrieved_docs,
            "answer": answer,
            "latency": round(latency, 4),
        }
