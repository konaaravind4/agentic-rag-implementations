"""
taxonomy/multi_agent_rag.py
===========================
Multi-Agent Agentic RAG (§5.2)
-------------------------------
Architecture from: "Agentic RAG: A Survey" (arXiv:2501.09136)

In Multi-Agent Agentic RAG, a **Coordinator Agent** dispatches the query
to several **Specialized Agents** that operate *concurrently* (via threads).
Each specialized agent independently retrieves information from its own data
source.  Once all agents have responded, the Coordinator synthesizes the
individual results into a single, unified answer.

Specialized Agents
------------------
1. ``SQLAgent``            – structured queries over an in-memory SQLite DB
2. ``SemanticAgent``       – FAISS dense-vector similarity search
3. ``WebAgent``            – simulated web-search corpus
4. ``RecommendationAgent`` – cosine-similarity content-based recommendations

The concurrent execution is handled via ``concurrent.futures.ThreadPoolExecutor``
so the wall-clock latency is roughly that of the *slowest* single agent rather
than the sum of all agents.
"""

from __future__ import annotations

import logging
import math
import re
import sqlite3
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import Any

from core.embeddings import Embedder          # noqa: F401
from core.vector_store import FAISSVectorStore
from core.llm import LocalLLM

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Shared demo corpus (used by agents that don't own a live data source)
# ---------------------------------------------------------------------------
_DEMO_CORPUS: list[dict] = [
    {"id": 1, "text": "Machine learning models require large amounts of training data.", "topic": "ml"},
    {"id": 2, "text": "Neural networks are inspired by the human brain structure.", "topic": "ml"},
    {"id": 3, "text": "Deep learning has revolutionised computer vision tasks.", "topic": "ml"},
    {"id": 4, "text": "The stock market showed significant gains this quarter.", "topic": "finance"},
    {"id": 5, "text": "Interest rates affect mortgage payments and home buying.", "topic": "finance"},
    {"id": 6, "text": "Cryptocurrency markets remain highly volatile.", "topic": "finance"},
    {"id": 7, "text": "Climate change impacts global weather patterns dramatically.", "topic": "science"},
    {"id": 8, "text": "Renewable energy sources include solar and wind power.", "topic": "science"},
    {"id": 9, "text": "Healthy diet includes vegetables, fruits, and lean proteins.", "topic": "health"},
    {"id": 10, "text": "Regular exercise improves cardiovascular health significantly.", "topic": "health"},
]

# Simulated web search corpus (fixed, in lieu of a live API)
_WEB_CORPUS: list[dict] = [
    {"url": "https://example.com/ai-news", "text": "AI research is advancing rapidly with new LLM architectures.", "domain": "technology"},
    {"url": "https://example.com/finance-update", "text": "Central banks are adjusting monetary policy in response to inflation.", "domain": "finance"},
    {"url": "https://example.com/health-science", "text": "New research links gut microbiome health to mental well-being.", "domain": "health"},
    {"url": "https://example.com/climate-report", "text": "Global temperatures have risen by 1.2°C since pre-industrial times.", "domain": "science"},
    {"url": "https://example.com/tech-market", "text": "Cloud computing spending continues to grow quarter over quarter.", "domain": "technology"},
    {"url": "https://example.com/medical-trial", "text": "A new drug trial shows promise for treating Alzheimer's disease.", "domain": "health"},
    {"url": "https://example.com/renewable-energy", "text": "Solar panel efficiency has improved by 30% in the past decade.", "domain": "science"},
    {"url": "https://example.com/crypto-analysis", "text": "Bitcoin adoption increases among institutional investors worldwide.", "domain": "finance"},
]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _term_vector(text: str) -> dict[str, float]:
    """Return a normalized term-frequency vector for the given text."""
    tokens = re.findall(r"\w+", text.lower())
    counts = Counter(tokens)
    total = sum(counts.values()) or 1
    return {t: c / total for t, c in counts.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two term-frequency vectors."""
    dot = sum(a.get(t, 0.0) * b.get(t, 0.0) for t in b)
    mag_a = math.sqrt(sum(v ** 2 for v in a.values())) or 1e-9
    mag_b = math.sqrt(sum(v ** 2 for v in b.values())) or 1e-9
    return dot / (mag_a * mag_b)


def _init_sqlite() -> sqlite3.Connection:
    """Create and populate an in-memory SQLite database from the demo corpus."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE documents (id INTEGER PRIMARY KEY, text TEXT, topic TEXT)"
    )
    for doc in _DEMO_CORPUS:
        cur.execute("INSERT INTO documents VALUES (?, ?, ?)", (doc["id"], doc["text"], doc["topic"]))
    conn.commit()
    return conn


# ===========================================================================
# Specialized Agents
# ===========================================================================

class SQLAgent:
    """
    Retrieves structured knowledge from an in-memory SQLite database.

    Uses keyword-based LIKE predicates to locate relevant rows.  In a
    production deployment this agent would generate proper SQL via an LLM
    or a text-to-SQL model.

    Parameters
    ----------
    db_conn : sqlite3.Connection
        Shared (thread-safe) SQLite connection.
    k : int
        Maximum number of results to return.
    """

    name = "SQLAgent"

    def __init__(self, db_conn: sqlite3.Connection, k: int = 3) -> None:
        self.db_conn = db_conn
        self.k = k

    def retrieve(self, query: str) -> dict[str, Any]:
        """
        Execute a keyword LIKE query against the SQLite documents table.

        Parameters
        ----------
        query : str
            Natural-language query; keywords are extracted automatically.

        Returns
        -------
        dict
            ``{"agent": name, "results": list[dict], "latency": float}``
        """
        t0 = time.perf_counter()
        logger.info("[%s] Retrieving for query=%r", self.name, query)

        keywords = re.findall(r"\w{3,}", query.lower())
        cur = self.db_conn.cursor()
        results: list[dict] = []
        seen: set[int] = set()

        for kw in (keywords or [""]):
            cur.execute(
                "SELECT id, text, topic FROM documents WHERE lower(text) LIKE ? LIMIT ?",
                (f"%{kw}%", self.k),
            )
            for row in cur.fetchall():
                doc_id, text, topic = row
                if doc_id not in seen:
                    seen.add(doc_id)
                    results.append({"id": doc_id, "text": text, "topic": topic, "source": "sqlite"})
            if len(results) >= self.k:
                break

        elapsed = time.perf_counter() - t0
        logger.info("[%s] Found %d results in %.3f s", self.name, len(results), elapsed)
        return {"agent": self.name, "results": results[: self.k], "latency": round(elapsed, 4)}


class SemanticAgent:
    """
    Retrieves documents using dense FAISS semantic vector search.

    Parameters
    ----------
    vector_store : FAISSVectorStore
        Pre-built FAISS index.
    k : int
        Number of top documents to retrieve.
    """

    name = "SemanticAgent"

    def __init__(self, vector_store: FAISSVectorStore, k: int = 3) -> None:
        self.vector_store = vector_store
        self.k = k

    def retrieve(self, query: str) -> dict[str, Any]:
        """
        Perform dense similarity search via FAISS.

        Parameters
        ----------
        query : str
            Natural-language query to embed and search.

        Returns
        -------
        dict
            ``{"agent": name, "results": list[dict], "latency": float}``
        """
        t0 = time.perf_counter()
        logger.info("[%s] Retrieving for query=%r", self.name, query)

        raw = self.vector_store.search(query, top_k=self.k)
        results = [
            {
                "text": item.get("text", item.get("content", str(item))),
                "score": float(item.get("score", item.get("similarity", 0.0))),
                "source": "faiss",
            }
            for item in raw
        ]

        elapsed = time.perf_counter() - t0
        logger.info("[%s] Found %d results in %.3f s", self.name, len(results), elapsed)
        return {"agent": self.name, "results": results, "latency": round(elapsed, 4)}


class WebAgent:
    """
    Simulates a web search by ranking a fixed corpus using cosine similarity.

    In a real deployment this agent would call a search API (e.g., Bing or
    Google Custom Search) and parse result snippets.

    Parameters
    ----------
    k : int
        Number of top web results to return.
    """

    name = "WebAgent"

    def __init__(self, k: int = 3) -> None:
        self.k = k

    def retrieve(self, query: str) -> dict[str, Any]:
        """
        Rank the simulated web corpus against the query via cosine similarity.

        Parameters
        ----------
        query : str
            The search query.

        Returns
        -------
        dict
            ``{"agent": name, "results": list[dict], "latency": float}``
        """
        t0 = time.perf_counter()
        logger.info("[%s] Simulating web search for query=%r", self.name, query)

        q_vec = _term_vector(query)
        scored = []
        for page in _WEB_CORPUS:
            p_vec = _term_vector(page["text"])
            sim = _cosine(q_vec, p_vec)
            scored.append(
                {
                    "url": page["url"],
                    "text": page["text"],
                    "domain": page["domain"],
                    "score": sim,
                    "source": "simulated_web",
                }
            )
        scored.sort(key=lambda x: x["score"], reverse=True)

        elapsed = time.perf_counter() - t0
        logger.info("[%s] Found %d results in %.3f s", self.name, self.k, elapsed)
        return {"agent": self.name, "results": scored[: self.k], "latency": round(elapsed, 4)}


class RecommendationAgent:
    """
    Content-based recommendation agent that returns demo corpus documents
    most similar to the query using TF cosine similarity.

    Parameters
    ----------
    k : int
        Number of recommendations to return.
    """

    name = "RecommendationAgent"

    def __init__(self, k: int = 3) -> None:
        self.k = k

    def retrieve(self, query: str) -> dict[str, Any]:
        """
        Recommend documents from the shared demo corpus via cosine similarity.

        Parameters
        ----------
        query : str
            Content description or user query.

        Returns
        -------
        dict
            ``{"agent": name, "results": list[dict], "latency": float}``
        """
        t0 = time.perf_counter()
        logger.info("[%s] Generating recommendations for query=%r", self.name, query)

        q_vec = _term_vector(query)
        scored = []
        for doc in _DEMO_CORPUS:
            d_vec = _term_vector(doc["text"])
            sim = _cosine(q_vec, d_vec)
            scored.append(
                {
                    "text": doc["text"],
                    "topic": doc["topic"],
                    "score": sim,
                    "source": "recommendation_engine",
                }
            )
        scored.sort(key=lambda x: x["score"], reverse=True)

        elapsed = time.perf_counter() - t0
        logger.info("[%s] Found %d recommendations in %.3f s", self.name, self.k, elapsed)
        return {"agent": self.name, "results": scored[: self.k], "latency": round(elapsed, 4)}


# ===========================================================================
# Coordinator Agent
# ===========================================================================

class CoordinatorAgent:
    """
    Dispatches queries to all specialized agents concurrently and
    synthesizes their results into a unified answer.

    Parameters
    ----------
    llm : LocalLLM
        Language model used for final synthesis.
    agents : list
        List of specialized agent instances (each must implement ``.retrieve()``).
    max_workers : int
        Number of threads in the executor pool (default 4).
    """

    def __init__(self, llm: LocalLLM, agents: list, max_workers: int = 4) -> None:
        self.llm = llm
        self.agents = agents
        self.max_workers = max_workers
        logger.info(
            "CoordinatorAgent initialised with %d agents: %s",
            len(agents),
            [a.name for a in agents],
        )

    def dispatch(self, query: str) -> dict[str, dict]:
        """
        Run all specialized agents concurrently and collect their results.

        Parameters
        ----------
        query : str
            The user query forwarded to every agent.

        Returns
        -------
        dict[str, dict]
            Mapping from agent name → agent result dict.
        """
        logger.info("CoordinatorAgent dispatching query to %d agents in parallel.", len(self.agents))
        agent_results: dict[str, dict] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_agent: dict[Future, Any] = {
                executor.submit(agent.retrieve, query): agent for agent in self.agents
            }
            for future in as_completed(future_to_agent):
                agent = future_to_agent[future]
                try:
                    result = future.result()
                    agent_results[agent.name] = result
                    logger.info(
                        "Agent %r finished: %d docs, %.3f s",
                        agent.name,
                        len(result.get("results", [])),
                        result.get("latency", 0.0),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error("Agent %r raised an exception: %s", agent.name, exc)
                    agent_results[agent.name] = {"agent": agent.name, "results": [], "error": str(exc)}

        return agent_results

    def synthesize(self, query: str, agent_results: dict[str, dict]) -> str:
        """
        Synthesize a single coherent answer from all agent results.

        Parameters
        ----------
        query : str
            Original user question.
        agent_results : dict[str, dict]
            All agent outputs keyed by agent name.

        Returns
        -------
        str
            The LLM-generated synthesized answer.
        """
        context_sections: list[str] = []
        for agent_name, result in agent_results.items():
            docs = result.get("results", [])
            if not docs:
                continue
            snippets = "\n".join(f"  • {d.get('text', '')}" for d in docs)
            context_sections.append(f"[{agent_name}]\n{snippets}")

        context_str = "\n\n".join(context_sections) or "No context retrieved."

        prompt = (
            "You are an expert assistant synthesizing information from multiple sources.\n"
            "Use ALL provided context to give a comprehensive, well-rounded answer.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {query}\n\n"
            "Synthesized Answer:"
        )
        return self.llm.generate(prompt)


# ===========================================================================
# Public facade: MultiAgentRAG
# ===========================================================================

class MultiAgentRAG:
    """
    Multi-Agent Agentic RAG (§5.2).

    A ``CoordinatorAgent`` fans the query out to four specialized agents
    (SQL, Semantic, Web, Recommendation) running concurrently in a thread
    pool.  After all agents complete, the Coordinator synthesizes their
    individual results into a single answer.

    Parameters
    ----------
    llm : LocalLLM
        Language model for synthesis.
    vector_store : FAISSVectorStore
        Pre-built FAISS index for the ``SemanticAgent``.
    k : int
        Number of documents each agent should retrieve (default 3).
    max_workers : int
        Thread-pool size for parallel agent execution (default 4).
    """

    def __init__(
        self,
        llm: LocalLLM,
        vector_store: FAISSVectorStore,
        k: int = 3,
        max_workers: int = 4,
    ) -> None:
        self.llm = llm
        self.vector_store = vector_store
        self.k = k

        # Shared SQLite connection (thread-safe for reads)
        db_conn = _init_sqlite()

        # Instantiate all specialized agents
        self._agents = [
            SQLAgent(db_conn=db_conn, k=k),
            SemanticAgent(vector_store=vector_store, k=k),
            WebAgent(k=k),
            RecommendationAgent(k=k),
        ]

        # Coordinator wraps the agents
        self._coordinator = CoordinatorAgent(
            llm=llm, agents=self._agents, max_workers=max_workers
        )
        logger.info("MultiAgentRAG initialised (k=%d, max_workers=%d)", k, max_workers)

    def query(self, question: str) -> dict:
        """
        Dispatch the question to all agents concurrently and synthesize results.

        Parameters
        ----------
        question : str
            The user's natural-language question.

        Returns
        -------
        dict
            A result dictionary with the following keys:

            * ``agent_results``     – per-agent retrieval outputs (dict of dicts).
            * ``synthesized_answer``– the Coordinator's unified answer.
            * ``latency``           – total wall-clock latency in seconds.
        """
        logger.info("=== MultiAgentRAG.query | question=%r ===", question)
        t_start = time.perf_counter()

        # 1. Dispatch to all specialized agents concurrently
        agent_results = self._coordinator.dispatch(question)

        # 2. Synthesize a unified answer
        synthesized_answer = self._coordinator.synthesize(question, agent_results)

        latency = time.perf_counter() - t_start
        logger.info("MultiAgentRAG.query completed in %.3f s", latency)

        return {
            "agent_results": agent_results,
            "synthesized_answer": synthesized_answer,
            "latency": round(latency, 4),
        }
