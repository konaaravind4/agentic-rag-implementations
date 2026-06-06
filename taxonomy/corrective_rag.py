"""
taxonomy/corrective_rag.py
==========================
Agentic Corrective RAG (CRAG) (§5.4)
--------------------------------------
Architecture from: "Agentic RAG: A Survey" (arXiv:2501.09136)

CRAG introduces a **self-correcting retrieval loop** that evaluates the
relevance of initially retrieved documents and, if they are deemed
insufficient, actively refines the query and fetches supplementary evidence
from an external source before generating the final answer.

5-Agent Pipeline
----------------
1. ``ContextRetrievalAgent``   – Retrieves an initial candidate set from FAISS.
2. ``RelevanceEvaluationAgent``– Scores each document on a 0–1 scale and flags
                                  those below ``relevance_threshold`` (default 0.5).
3. ``QueryRefinementAgent``    – Rewrites the query when average relevance is too
                                  low, preserving intent while improving precision.
4. ``ExternalKnowledgeAgent``  – Simulates web search; called when the local
                                  context is insufficient.
5. ``ResponseSynthesisAgent``  – Generates the final answer from the best
                                  available context.

Correction Loop
---------------
The pipeline iterates up to ``max_iterations`` times (default 2).  On each
iteration:

* If avg(relevance_scores) >= ``relevance_threshold`` → proceed to synthesis.
* Otherwise → QueryRefinementAgent rewrites the query → ExternalKnowledgeAgent
  fetches supplementary docs → retry with augmented context.

This mirrors the CRAG mechanism described in §5.4 of the survey.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter
from typing import Any

from core.vector_store import FAISSVectorStore
from core.llm import LocalLLM

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Simulated external (web) knowledge corpus
# ---------------------------------------------------------------------------
_EXTERNAL_CORPUS: list[dict] = [
    {"url": "https://news.example.com/ai-2024", "text": "Large language models are transforming AI applications in 2024.", "domain": "technology"},
    {"url": "https://health.example.com/diabetes", "text": "Type 2 diabetes can be managed through diet, exercise, and medication.", "domain": "health"},
    {"url": "https://finance.example.com/rates", "text": "The Federal Reserve raised interest rates to combat persistent inflation.", "domain": "finance"},
    {"url": "https://science.example.com/climate", "text": "Arctic ice loss accelerates due to rising global temperatures.", "domain": "science"},
    {"url": "https://tech.example.com/quantum", "text": "Quantum computing promises exponential speedups for certain algorithms.", "domain": "technology"},
    {"url": "https://med.example.com/vaccines", "text": "mRNA vaccine technology opens new avenues for cancer immunotherapy.", "domain": "health"},
    {"url": "https://econ.example.com/gdp", "text": "GDP growth slowed in Q3 amid supply chain disruptions.", "domain": "finance"},
    {"url": "https://env.example.com/solar", "text": "Solar energy capacity doubled globally over the past five years.", "domain": "science"},
    {"url": "https://edu.example.com/learning", "text": "Adaptive learning systems personalise education using AI algorithms.", "domain": "technology"},
    {"url": "https://bio.example.com/crispr", "text": "CRISPR gene editing enables precise modification of DNA sequences.", "domain": "health"},
]

# ---------------------------------------------------------------------------
# Relevance evaluation prompt
# ---------------------------------------------------------------------------
_RELEVANCE_PROMPT = """You are a relevance evaluator for a RAG system.
Score how relevant the given document is to the user's question.

Question: "{query}"
Document: "{doc_text}"

Return a JSON object ONLY:
{{
  "score": <float between 0.0 and 1.0>,
  "justification": "<one sentence>"
}}
A score of 1.0 means perfectly relevant; 0.0 means completely irrelevant.
"""

# ---------------------------------------------------------------------------
# Query refinement prompt
# ---------------------------------------------------------------------------
_REFINEMENT_PROMPT = """You are a query refinement specialist for a RAG system.
The initial query retrieved documents with low relevance scores.
Rewrite the query to be more specific and precise while preserving the original intent.

Original query: "{query}"
Average relevance of retrieved docs: {avg_relevance:.2f} (threshold: {threshold:.2f})

Return a JSON object ONLY:
{{
  "refined_query": "<improved query>",
  "changes_made": "<brief description of changes>"
}}
"""

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _term_vector(text: str) -> dict[str, float]:
    """Compute a normalized term-frequency vector for text."""
    tokens = re.findall(r"\w+", text.lower())
    counts = Counter(tokens)
    total = sum(counts.values()) or 1
    return {t: c / total for t, c in counts.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two TF vectors."""
    dot = sum(a.get(t, 0.0) * b.get(t, 0.0) for t in b)
    mag_a = math.sqrt(sum(v ** 2 for v in a.values())) or 1e-9
    mag_b = math.sqrt(sum(v ** 2 for v in b.values())) or 1e-9
    return dot / (mag_a * mag_b)


# ===========================================================================
# Agent 1: ContextRetrievalAgent
# ===========================================================================

class ContextRetrievalAgent:
    """
    Agent 1 – Initial context retrieval from the FAISS vector store.

    Parameters
    ----------
    vector_store : FAISSVectorStore
        Pre-built dense index.
    k : int
        Number of candidate documents to retrieve.
    """

    name = "ContextRetrievalAgent"

    def __init__(self, vector_store: FAISSVectorStore, k: int = 5) -> None:
        self.vector_store = vector_store
        self.k = k

    def retrieve(self, query: str) -> list[dict]:
        """
        Retrieve the top-k candidate documents for the query.

        Parameters
        ----------
        query : str
            User query (original or refined).

        Returns
        -------
        list[dict]
            Retrieved documents, each with keys ``text``, ``score``, ``source``.
        """
        logger.info("[%s] Retrieving context for query=%r", self.name, query)
        t0 = time.perf_counter()
        raw = self.vector_store.search(query, top_k=self.k)
        docs = [
            {
                "text": item.get("text", item.get("content", str(item))),
                "score": float(item.get("score", item.get("similarity", 0.0))),
                "source": "faiss",
            }
            for item in raw
        ]
        elapsed = time.perf_counter() - t0
        logger.info("[%s] Retrieved %d docs in %.3f s", self.name, len(docs), elapsed)
        return docs


# ===========================================================================
# Agent 2: RelevanceEvaluationAgent
# ===========================================================================

class RelevanceEvaluationAgent:
    """
    Agent 2 – Evaluates the relevance of each retrieved document.

    Each document is scored on a 0–1 scale using the LLM.  Documents below
    ``relevance_threshold`` are flagged.  The agent also computes an
    aggregate average score used by the correction loop.

    Parameters
    ----------
    llm : LocalLLM
        Language model for relevance scoring.
    relevance_threshold : float
        Minimum acceptable relevance score (default 0.5).
    """

    name = "RelevanceEvaluationAgent"

    def __init__(self, llm: LocalLLM, relevance_threshold: float = 0.5) -> None:
        self.llm = llm
        self.relevance_threshold = relevance_threshold

    def _score_document(self, query: str, doc: dict) -> dict[str, Any]:
        """
        Ask the LLM to score a single document's relevance.

        Uses a structured prompt; falls back to a heuristic cosine score
        if the LLM response cannot be parsed.

        Parameters
        ----------
        query : str
            User query.
        doc : dict
            Document dict with at least a ``text`` key.

        Returns
        -------
        dict
            Extended document dict with ``relevance_score``, ``above_threshold``,
            and ``justification`` fields added.
        """
        import json

        prompt = _RELEVANCE_PROMPT.format(query=query, doc_text=doc.get("text", "")[:500])
        raw = self.llm.generate(prompt)

        try:
            json_match = re.search(r"\{.*?\}", raw, re.DOTALL)
            parsed = json.loads(json_match.group() if json_match else raw)
            rel_score = max(0.0, min(1.0, float(parsed.get("score", 0.0))))
            justification = parsed.get("justification", "")
        except Exception:  # noqa: BLE001
            # Fallback: cosine similarity between query and doc text
            q_vec = _term_vector(query)
            d_vec = _term_vector(doc.get("text", ""))
            rel_score = _cosine(q_vec, d_vec)
            justification = "Fallback: cosine similarity used."

        return {
            **doc,
            "relevance_score": round(rel_score, 4),
            "above_threshold": rel_score >= self.relevance_threshold,
            "justification": justification,
        }

    def evaluate(self, query: str, docs: list[dict]) -> tuple[list[dict], float]:
        """
        Score all retrieved documents and return an aggregate summary.

        Parameters
        ----------
        query : str
            User query.
        docs : list[dict]
            Candidate documents from ``ContextRetrievalAgent``.

        Returns
        -------
        tuple[list[dict], float]
            ``(scored_docs, avg_relevance)`` where each doc in ``scored_docs``
            has ``relevance_score``, ``above_threshold``, and ``justification``
            fields appended.
        """
        logger.info("[%s] Evaluating %d documents for query=%r", self.name, len(docs), query)
        scored: list[dict] = []
        for doc in docs:
            scored_doc = self._score_document(query, doc)
            logger.debug(
                "[%s] Doc relevance=%.3f above_threshold=%s | text=%r",
                self.name,
                scored_doc["relevance_score"],
                scored_doc["above_threshold"],
                scored_doc["text"][:60],
            )
            scored.append(scored_doc)

        avg = sum(d["relevance_score"] for d in scored) / max(len(scored), 1)
        logger.info(
            "[%s] Avg relevance=%.3f (threshold=%.2f) | %d/%d docs above threshold",
            self.name, avg, self.relevance_threshold,
            sum(1 for d in scored if d["above_threshold"]), len(scored),
        )
        return scored, round(avg, 4)


# ===========================================================================
# Agent 3: QueryRefinementAgent
# ===========================================================================

class QueryRefinementAgent:
    """
    Agent 3 – Rewrites the query when retrieved context is insufficient.

    Triggered when ``avg_relevance < relevance_threshold``.  The LLM is
    prompted to produce a more specific and precise version of the original
    query that is more likely to retrieve relevant documents.

    Parameters
    ----------
    llm : LocalLLM
        Language model for query rewriting.
    relevance_threshold : float
        Threshold that triggered the refinement (used in the prompt).
    """

    name = "QueryRefinementAgent"

    def __init__(self, llm: LocalLLM, relevance_threshold: float = 0.5) -> None:
        self.llm = llm
        self.relevance_threshold = relevance_threshold

    def refine(self, query: str, avg_relevance: float) -> tuple[str, str]:
        """
        Rewrite the query to improve retrieval relevance.

        Parameters
        ----------
        query : str
            Original (or previously refined) query.
        avg_relevance : float
            Current average relevance score that triggered the call.

        Returns
        -------
        tuple[str, str]
            ``(refined_query, changes_description)``
        """
        import json

        logger.info(
            "[%s] Refining query=%r | avg_relevance=%.3f", self.name, query, avg_relevance
        )
        prompt = _REFINEMENT_PROMPT.format(
            query=query,
            avg_relevance=avg_relevance,
            threshold=self.relevance_threshold,
        )
        raw = self.llm.generate(prompt)

        try:
            json_match = re.search(r"\{.*?\}", raw, re.DOTALL)
            parsed = json.loads(json_match.group() if json_match else raw)
            refined_query = parsed.get("refined_query", query).strip()
            changes = parsed.get("changes_made", "")
        except Exception:  # noqa: BLE001
            # Fallback: append clarification suffix
            refined_query = query + " (provide detailed explanation)"
            changes = "Fallback: appended clarification suffix."

        if not refined_query:
            refined_query = query + " detailed"

        logger.info("[%s] Refined query=%r | changes=%r", self.name, refined_query, changes)
        return refined_query, changes


# ===========================================================================
# Agent 4: ExternalKnowledgeAgent
# ===========================================================================

class ExternalKnowledgeAgent:
    """
    Agent 4 – Fetches supplementary knowledge from a simulated external source.

    Called when the local context is deemed insufficient by the correction
    loop.  In production this agent would issue HTTP requests to a search
    API; here it ranks a fixed corpus by cosine similarity.

    Parameters
    ----------
    k : int
        Number of external documents to return.
    """

    name = "ExternalKnowledgeAgent"

    def __init__(self, k: int = 3) -> None:
        self.k = k

    def fetch(self, query: str) -> list[dict]:
        """
        Return the top-k external documents most relevant to the query.

        Parameters
        ----------
        query : str
            The (possibly refined) user query.

        Returns
        -------
        list[dict]
            External documents with keys ``url``, ``text``, ``domain``,
            ``score``, ``source``.
        """
        logger.info("[%s] Fetching external knowledge for query=%r", self.name, query)
        t0 = time.perf_counter()

        q_vec = _term_vector(query)
        scored = []
        for page in _EXTERNAL_CORPUS:
            p_vec = _term_vector(page["text"])
            sim = _cosine(q_vec, p_vec)
            scored.append(
                {
                    "url": page["url"],
                    "text": page["text"],
                    "domain": page["domain"],
                    "score": sim,
                    "source": "external_web",
                }
            )
        scored.sort(key=lambda x: x["score"], reverse=True)
        results = scored[: self.k]

        elapsed = time.perf_counter() - t0
        logger.info(
            "[%s] Fetched %d external docs in %.3f s", self.name, len(results), elapsed
        )
        return results


# ===========================================================================
# Agent 5: ResponseSynthesisAgent
# ===========================================================================

class ResponseSynthesisAgent:
    """
    Agent 5 – Generates the final answer from the best available context.

    Receives all collected context (local + external) and synthesizes a
    coherent, grounded answer using the LLM.

    Parameters
    ----------
    llm : LocalLLM
        Language model for answer generation.
    """

    name = "ResponseSynthesisAgent"

    def __init__(self, llm: LocalLLM) -> None:
        self.llm = llm

    def synthesize(
        self,
        query: str,
        local_docs: list[dict],
        external_docs: list[dict],
        num_iterations: int,
    ) -> str:
        """
        Produce the final answer from local + external documents.

        Only documents above the relevance threshold (``above_threshold=True``)
        are used from the local pool to keep context clean.

        Parameters
        ----------
        query : str
            Original (or last refined) user question.
        local_docs : list[dict]
            Scored local documents (may include ``above_threshold`` flag).
        external_docs : list[dict]
            Supplementary documents from ``ExternalKnowledgeAgent``.
        num_iterations : int
            Number of correction loop iterations performed (for transparency).

        Returns
        -------
        str
            The generated answer.
        """
        logger.info(
            "[%s] Synthesizing answer | local=%d external=%d iterations=%d",
            self.name, len(local_docs), len(external_docs), num_iterations,
        )

        # Prefer docs above threshold; fall back to all docs if none pass
        above = [d for d in local_docs if d.get("above_threshold", True)]
        context_local = above if above else local_docs

        local_snippets = "\n".join(
            f"  [Local-{i+1}] {d.get('text', '')} (relevance={d.get('relevance_score', 'N/A')})"
            for i, d in enumerate(context_local)
        )
        ext_snippets = "\n".join(
            f"  [Ext-{i+1}] {d.get('text', '')} | src={d.get('url', 'web')}"
            for i, d in enumerate(external_docs)
        )

        context_str = ""
        if local_snippets:
            context_str += f"Local Context:\n{local_snippets}\n\n"
        if ext_snippets:
            context_str += f"External Context:\n{ext_snippets}\n\n"
        if not context_str:
            context_str = "No relevant context was retrieved.\n"

        prompt = (
            f"You are a precise and helpful assistant (CRAG system, {num_iterations} retrieval"
            f" iteration(s) performed).\n"
            "Answer the question using ONLY the provided context.  If context is insufficient,"
            " clearly state the limitation.\n\n"
            f"{context_str}"
            f"Question: {query}\n\n"
            "Answer:"
        )
        return self.llm.generate(prompt)


# ===========================================================================
# Public facade: CorrectiveRAG
# ===========================================================================

class CorrectiveRAG:
    """
    Agentic Corrective RAG (CRAG) (§5.4).

    Implements the 5-agent self-correcting pipeline described in the survey.
    The correction loop re-tries retrieval with a refined query and external
    knowledge augmentation whenever the average relevance score of locally
    retrieved documents falls below ``relevance_threshold``.

    Parameters
    ----------
    llm : LocalLLM
        Language model shared across evaluation, refinement, and synthesis agents.
    vector_store : FAISSVectorStore
        FAISS index for local context retrieval.
    k : int
        Number of documents to retrieve per iteration (default 5).
    relevance_threshold : float
        Minimum average relevance required to skip correction (default 0.5).
    max_iterations : int
        Maximum number of correction loop iterations (default 2).
    """

    def __init__(
        self,
        llm: LocalLLM,
        vector_store: FAISSVectorStore,
        k: int = 5,
        relevance_threshold: float = 0.5,
        max_iterations: int = 2,
    ) -> None:
        self.llm = llm
        self.vector_store = vector_store
        self.k = k
        self.relevance_threshold = relevance_threshold
        self.max_iterations = max_iterations

        # Instantiate all 5 pipeline agents
        self._retrieval_agent = ContextRetrievalAgent(vector_store=vector_store, k=k)
        self._relevance_agent = RelevanceEvaluationAgent(
            llm=llm, relevance_threshold=relevance_threshold
        )
        self._refinement_agent = QueryRefinementAgent(
            llm=llm, relevance_threshold=relevance_threshold
        )
        self._external_agent = ExternalKnowledgeAgent(k=k)
        self._synthesis_agent = ResponseSynthesisAgent(llm=llm)

        logger.info(
            "CorrectiveRAG initialised | k=%d threshold=%.2f max_iter=%d",
            k, relevance_threshold, max_iterations,
        )

    def query(self, question: str) -> dict:
        """
        Run the full CRAG pipeline with iterative self-correction.

        Algorithm
        ---------
        1. Retrieve initial context (Agent 1).
        2. Evaluate document relevance (Agent 2).
        3. If avg relevance >= threshold → go to step 6.
        4. Refine query (Agent 3) + fetch external knowledge (Agent 4).
        5. Repeat from step 2 up to ``max_iterations`` times.
        6. Synthesize final answer (Agent 5).

        Parameters
        ----------
        question : str
            The user's natural-language question.

        Returns
        -------
        dict
            Result dictionary with keys:

            * ``retrieval_iterations`` – number of correction iterations performed.
            * ``relevance_scores``     – per-iteration avg relevance scores.
            * ``query_refinements``    – list of refined queries (one per iteration).
            * ``final_docs``           – documents used for final synthesis.
            * ``external_docs``        – external documents fetched (if any).
            * ``final_answer``         – the synthesized answer.
            * ``latency``              – total wall-clock time in seconds.
        """
        logger.info("=== CorrectiveRAG.query | question=%r ===", question)
        t_start = time.perf_counter()

        current_query = question
        all_relevance_scores: list[float] = []
        query_refinements: list[str] = []
        external_docs: list[dict] = []
        scored_docs: list[dict] = []
        iteration = 0

        # ----------------------------------------------------------------
        # Correction loop
        # ----------------------------------------------------------------
        for iteration in range(1, self.max_iterations + 1):
            logger.info(
                "--- CRAG iteration %d/%d | query=%r ---",
                iteration, self.max_iterations, current_query,
            )

            # Step 1: Retrieve
            raw_docs = self._retrieval_agent.retrieve(current_query)

            # Step 2: Evaluate relevance
            scored_docs, avg_relevance = self._relevance_agent.evaluate(current_query, raw_docs)
            all_relevance_scores.append(avg_relevance)

            logger.info(
                "Iteration %d | avg_relevance=%.3f (threshold=%.2f)",
                iteration, avg_relevance, self.relevance_threshold,
            )

            if avg_relevance >= self.relevance_threshold:
                logger.info(
                    "Relevance threshold met at iteration %d. Proceeding to synthesis.", iteration
                )
                break

            # Threshold not met — correct if we have iterations remaining
            if iteration < self.max_iterations:
                # Step 3: Refine query
                refined_query, changes = self._refinement_agent.refine(
                    current_query, avg_relevance
                )
                query_refinements.append(refined_query)
                current_query = refined_query

                # Step 4: Fetch external knowledge (accumulate across iterations)
                new_external = self._external_agent.fetch(current_query)
                external_docs.extend(new_external)
                # Deduplicate by URL or text
                seen_texts: set[str] = set()
                unique_external: list[dict] = []
                for ext in external_docs:
                    key = ext.get("url", ext.get("text", ""))
                    if key not in seen_texts:
                        seen_texts.add(key)
                        unique_external.append(ext)
                external_docs = unique_external
            else:
                # Max iterations reached; still fetch external as fallback
                logger.info(
                    "Max iterations reached. Fetching external knowledge as fallback."
                )
                fallback_external = self._external_agent.fetch(current_query)
                for ext in fallback_external:
                    key = ext.get("url", ext.get("text", ""))
                    if not any(
                        e.get("url", e.get("text", "")) == key for e in external_docs
                    ):
                        external_docs.append(ext)

        # ----------------------------------------------------------------
        # Step 5 / Agent 5: Synthesize final answer
        # ----------------------------------------------------------------
        final_answer = self._synthesis_agent.synthesize(
            query=question,         # always use the ORIGINAL question for synthesis
            local_docs=scored_docs,
            external_docs=external_docs,
            num_iterations=iteration,
        )

        latency = time.perf_counter() - t_start
        logger.info("CorrectiveRAG.query completed in %.3f s | iterations=%d", latency, iteration)

        return {
            "retrieval_iterations": iteration,
            "relevance_scores": all_relevance_scores,
            "query_refinements": query_refinements,
            "final_docs": scored_docs,
            "external_docs": external_docs,
            "final_answer": final_answer,
            "latency": round(latency, 4),
        }
