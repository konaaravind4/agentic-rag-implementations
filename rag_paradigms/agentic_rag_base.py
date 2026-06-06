"""
agentic_rag_base.py
===================
Implements the Abstract Base Agentic RAG class as described in §2.3.5 of:
  "Agentic RAG: A Survey" (arXiv:2501.09136)

Agentic RAG elevates the RAG pipeline from a passive, single-pass process
into an *active reasoning loop*: the agent decides *when* to retrieve,
*what* to retrieve, *whether to retrieve again*, and *when it has enough
information* to produce a reliable answer.

This module provides ``AgenticRAGBase`` — an abstract base class that:

  * Defines the public ``query(question: str) -> dict`` interface that all
    concrete agentic RAG implementations must implement.
  * Provides shared, reusable utility methods::

      _retrieve(query, k)                → list[dict]
      _generate(prompt)                  → str
      _evaluate_relevance(query, doc)    → float  (0.0–1.0)
      _should_retrieve_more(answer, query) → bool

  * Supplies a concrete ``_should_retrieve_more`` implementation based on
    LLM self-evaluation — the hallmark of agentic behaviour.
  * Offers ``_format_docs`` and ``_build_answer_prompt`` helpers so
    subclasses do not need to repeat boilerplate.

Subclass example
----------------
See ``rag_paradigms/taxonomy/`` for concrete implementations:
  * ``SingleAgentRAG``  — a ReAct-style loop with tool use.
  * ``MultiAgentRAG``   — a coordinator + specialist agent topology.
  * ``HierarchicalRAG`` — a planner / retriever / reader hierarchy.
  * ``CorrectiveRAG``   — iterative retrieval with self-correction.
"""

from __future__ import annotations

import time
import logging
from abc import ABC, abstractmethod
from typing import Any

from core.embeddings import Embedder
from core.vector_store import FAISSVectorStore
from core.llm import LocalLLM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_RELEVANCE_PROMPT = (
    "On a scale from 0.0 to 1.0, rate how relevant the following document is "
    "to answering the query. Output only the float score, nothing else.\n\n"
    "Query: {query}\n\nDocument:\n{doc_text}\n\nRelevance score (0.0-1.0):"
)

_RETRIEVE_MORE_PROMPT = (
    "You are evaluating whether a generated answer adequately addresses a question.\n\n"
    "Question: {question}\n\n"
    "Generated answer: {answer}\n\n"
    "Does the answer:\n"
    "  1. Directly address the question?\n"
    "  2. Contain specific, factual information (not vague or evasive)?\n"
    "  3. Show no signs of hallucination or uncertainty?\n\n"
    "If ALL three criteria are met, respond 'SUFFICIENT'.\n"
    "If ANY criterion is not met, respond 'INSUFFICIENT'.\n\n"
    "Evaluation (SUFFICIENT or INSUFFICIENT):"
)

_ANSWER_PROMPT = "Context: {context}\n\nQuestion: {question}\n\nAnswer:"


# ---------------------------------------------------------------------------
# AgenticRAGBase — abstract base class
# ---------------------------------------------------------------------------

class AgenticRAGBase(ABC):
    """
    Abstract Base for Agentic RAG implementations (§2.3.5).

    Provides the shared retrieval, generation, relevance-evaluation, and
    self-evaluation utilities that all agentic RAG variants build upon.
    Concrete subclasses must implement the ``query()`` method, which defines
    the specific agentic reasoning loop (e.g., ReAct, MCTS, corrective RAG).

    Parameters
    ----------
    vector_store : FAISSVectorStore
        Pre-initialised FAISS vector store for document retrieval.
    llm : LocalLLM
        Language model used for generation, relevance scoring, and
        self-evaluation.
    k : int, optional
        Default number of documents to retrieve per retrieval call (default: 3).
    relevance_threshold : float, optional
        Minimum relevance score (0.0–1.0) for a document to be considered
        useful.  Documents below this threshold are filtered out in
        ``_retrieve``.  Set to ``0.0`` to disable filtering (default: 0.4).

    Notes
    -----
    All shared utilities are prefixed with a single underscore to signal
    that they are for subclass use, not public API consumers.
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        llm: LocalLLM,
        k: int = 3,
        relevance_threshold: float = 0.4,
    ) -> None:
        """
        Initialise the base agentic RAG agent.

        Parameters
        ----------
        vector_store : FAISSVectorStore
            Vector store for semantic document search.
        llm : LocalLLM
            Language model for all generation and evaluation tasks.
        k : int
            Default retrieval cardinality.
        relevance_threshold : float
            Minimum relevance score to accept a retrieved document.
            Valid range: [0.0, 1.0].
        """
        if not 0.0 <= relevance_threshold <= 1.0:
            raise ValueError(
                f"relevance_threshold must be in [0.0, 1.0], got {relevance_threshold}"
            )
        self.vector_store = vector_store
        self.llm = llm
        self.k = k
        self.relevance_threshold = relevance_threshold
        self.embedder: Embedder = Embedder()
        logger.info(
            "%s initialised (k=%d, relevance_threshold=%.2f)",
            self.__class__.__name__,
            k,
            relevance_threshold,
        )

    # ------------------------------------------------------------------
    # Abstract interface — subclasses MUST implement this
    # ------------------------------------------------------------------

    @abstractmethod
    def query(self, question: str) -> dict[str, Any]:
        """
        Execute the agentic RAG reasoning loop for a given question.

        Subclasses implement their specific agent topology and reasoning
        strategy here (e.g., ReAct loop, corrective retrieval, multi-agent
        delegation).

        Parameters
        ----------
        question : str
            The user's natural-language question.

        Returns
        -------
        dict
            At minimum, the result dict must contain:

            ``answer`` : str
                The final generated answer.
            ``latency`` : dict
                At minimum ``{"total": float}`` (wall-clock seconds).

            Implementations may include additional keys (e.g., ``"iterations"``,
            ``"retrieved_docs"``, ``"reasoning_trace"``).

        Raises
        ------
        ValueError
            If ``question`` is empty.
        """
        ...

    # ------------------------------------------------------------------
    # Shared utility: retrieve
    # ------------------------------------------------------------------

    def _retrieve(
        self,
        query: str,
        k: int | None = None,
        filter_by_relevance: bool = True,
    ) -> list[dict]:
        """
        Embed a query and retrieve the top-k documents from the vector store.

        Optionally filters retrieved documents by computing an LLM-based
        relevance score and discarding those below ``self.relevance_threshold``.

        Parameters
        ----------
        query : str
            The query string to embed and search with.
        k : int or None, optional
            Number of documents to retrieve.  Defaults to ``self.k``.
        filter_by_relevance : bool, optional
            If ``True`` (default), apply ``_evaluate_relevance`` to each
            retrieved document and drop those scoring below the threshold.
            Set to ``False`` for faster retrieval without LLM-based filtering.

        Returns
        -------
        list[dict]
            Retrieved documents, each containing at least ``"id"``,
            ``"text"``, ``"score"``, and ``"metadata"``.  If
            ``filter_by_relevance=True``, documents are additionally
            augmented with a ``"relevance_score"`` key.
        """
        effective_k = k if k is not None else self.k
        t0 = time.perf_counter()
        query_embedding = self.embedder.embed(query)
        docs = self.vector_store.search(query_embedding=query_embedding, k=effective_k)
        retrieval_time = time.perf_counter() - t0
        logger.debug(
            "_retrieve: retrieved %d docs for query '%.60s …' in %.4f s",
            len(docs), query, retrieval_time,
        )

        if filter_by_relevance and docs:
            filtered_docs: list[dict] = []
            for doc in docs:
                relevance = self._evaluate_relevance(query, doc)
                if relevance >= self.relevance_threshold:
                    filtered_docs.append({**doc, "relevance_score": relevance})
                else:
                    logger.debug(
                        "_retrieve: filtered doc (score=%.3f < threshold=%.3f): %.60s …",
                        relevance, self.relevance_threshold, doc.get("text", ""),
                    )
            logger.debug(
                "_retrieve: %d/%d docs passed relevance filter",
                len(filtered_docs), len(docs),
            )
            return filtered_docs

        return docs

    # ------------------------------------------------------------------
    # Shared utility: generate
    # ------------------------------------------------------------------

    def _generate(self, prompt: str) -> str:
        """
        Call the LLM to generate text from a prompt string.

        A thin wrapper around ``self.llm.generate`` that adds structured
        logging and a consistency point for subclasses.

        Parameters
        ----------
        prompt : str
            The full formatted prompt to send to the LLM.

        Returns
        -------
        str
            The LLM-generated text, stripped of leading/trailing whitespace.

        Raises
        ------
        ValueError
            If ``prompt`` is an empty string.
        """
        if not prompt.strip():
            raise ValueError("prompt must not be empty.")
        t0 = time.perf_counter()
        output = self.llm.generate(prompt).strip()
        logger.debug("_generate: %d chars in %.4f s", len(output), time.perf_counter() - t0)
        return output

    # ------------------------------------------------------------------
    # Shared utility: evaluate relevance
    # ------------------------------------------------------------------

    def _evaluate_relevance(self, query: str, doc: dict) -> float:
        """
        Score how relevant a retrieved document is to a given query.

        Asks the LLM to rate relevance on a continuous 0.0–1.0 scale.
        Falls back to 0.5 (neutral) if the LLM output cannot be parsed.

        Parameters
        ----------
        query : str
            The query string (or reformulated sub-query) being evaluated.
        doc : dict
            A retrieved document dict containing at least a ``"text"`` key.

        Returns
        -------
        float
            Relevance score in the range [0.0, 1.0].  Higher is more relevant.

        Notes
        -----
        This method incurs one LLM call per document.  In high-throughput
        settings, consider batching or using a lightweight cross-encoder
        instead.
        """
        doc_text = doc.get("text", "")
        if not doc_text.strip():
            logger.debug("_evaluate_relevance: empty document text; returning 0.0")
            return 0.0

        prompt = _RELEVANCE_PROMPT.format(query=query, doc_text=doc_text)
        raw = self.llm.generate(prompt).strip()

        try:
            # Accept formats like "0.8", ".8", "0.80", "8/10"
            if "/" in raw:
                num, denom = raw.split("/", 1)
                score = float(num.strip()) / float(denom.strip())
            else:
                # Extract first float-like token
                token = raw.split()[0].rstrip(".,;")
                score = float(token)
            score = max(0.0, min(1.0, score))
        except (ValueError, IndexError, ZeroDivisionError):
            logger.warning(
                "_evaluate_relevance: could not parse '%s'; defaulting to 0.5", raw
            )
            score = 0.5

        logger.debug(
            "_evaluate_relevance: score=%.3f for doc '%.60s …'", score, doc_text
        )
        return score

    # ------------------------------------------------------------------
    # Shared utility: should retrieve more (self-evaluation)
    # ------------------------------------------------------------------

    def _should_retrieve_more(self, answer: str, query: str) -> bool:
        """
        LLM self-evaluation: decide whether the current answer is sufficient.

        This is the core *agentic* behaviour — instead of always outputting
        the first answer, the agent evaluates its own response and decides
        whether to perform additional retrieval iterations.

        The LLM is prompted to assess three criteria:
          1. Does the answer directly address the question?
          2. Does it contain specific, factual information?
          3. Does it show no signs of hallucination or uncertainty?

        Parameters
        ----------
        answer : str
            The candidate answer generated so far.
        query : str
            The original user question.

        Returns
        -------
        bool
            ``True``  — the answer is INSUFFICIENT; more retrieval is needed.
            ``False`` — the answer is SUFFICIENT; stop the loop.

        Notes
        -----
        *Uncertainty phrases* are used as a secondary heuristic: if the LLM
        evaluation is ambiguous but the answer contains phrases like
        "I don't know" or "I'm not sure", the method returns ``True``
        regardless of the formal evaluation.
        """
        if not answer.strip():
            logger.debug("_should_retrieve_more: empty answer → need more retrieval.")
            return True

        # Primary: structured LLM self-evaluation
        prompt = _RETRIEVE_MORE_PROMPT.format(question=query, answer=answer)
        evaluation = self.llm.generate(prompt).strip().upper()

        # Parse the binary verdict
        if "INSUFFICIENT" in evaluation:
            logger.debug("_should_retrieve_more: LLM verdict=INSUFFICIENT → retrieve more.")
            return True
        if "SUFFICIENT" in evaluation:
            logger.debug("_should_retrieve_more: LLM verdict=SUFFICIENT → stop.")
            return False

        # Secondary: keyword heuristic fallback for ambiguous verdicts
        uncertainty_phrases = [
            "i don't know", "i do not know", "i'm not sure", "i am not sure",
            "unclear", "cannot determine", "insufficient information",
            "no information", "not enough", "i cannot",
        ]
        answer_lower = answer.lower()
        if any(phrase in answer_lower for phrase in uncertainty_phrases):
            logger.debug(
                "_should_retrieve_more: uncertainty phrase detected → retrieve more."
            )
            return True

        # If evaluation is ambiguous and no uncertainty detected, assume sufficient
        logger.debug(
            "_should_retrieve_more: ambiguous verdict '%s'; assuming SUFFICIENT.", evaluation
        )
        return False

    # ------------------------------------------------------------------
    # Shared utility: format docs
    # ------------------------------------------------------------------

    def _format_docs(self, docs: list[dict], include_scores: bool = False) -> str:
        """
        Serialise a list of retrieved documents into a readable context string.

        Parameters
        ----------
        docs : list[dict]
            Retrieved documents (each must contain a ``"text"`` key).
        include_scores : bool, optional
            If ``True``, append the ``"relevance_score"`` or ``"score"``
            value to each document header (useful for debugging).

        Returns
        -------
        str
            A numbered, newline-separated string ready for prompt insertion.
            Returns ``"No relevant documents found."`` when ``docs`` is empty.
        """
        if not docs:
            return "No relevant documents found."
        parts: list[str] = []
        for i, doc in enumerate(docs, 1):
            header = f"[{i}]"
            if include_scores:
                score = doc.get("relevance_score", doc.get("score", "N/A"))
                header += f" (relevance={score:.3f})" if isinstance(score, float) else f" (score={score})"
            parts.append(f"{header} {doc['text']}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Shared utility: build answer prompt
    # ------------------------------------------------------------------

    def _build_answer_prompt(self, context: str, question: str) -> str:
        """
        Construct the standard answer-generation prompt.

        Parameters
        ----------
        context : str
            Formatted context string (from ``_format_docs`` or a subclass
            custom builder).
        question : str
            The user's original question.

        Returns
        -------
        str
            A formatted prompt string ready to pass to ``_generate``.
        """
        return _ANSWER_PROMPT.format(context=context, question=question)

    # ------------------------------------------------------------------
    # Shared utility: measure latency context manager
    # ------------------------------------------------------------------

    class _Timer:
        """
        Lightweight context manager for measuring elapsed wall-clock time.

        Usage::

            timer = AgenticRAGBase._Timer()
            with timer:
                do_something()
            print(timer.elapsed)  # seconds as float
        """

        def __init__(self) -> None:
            self.elapsed: float = 0.0
            self._start: float = 0.0

        def __enter__(self) -> "AgenticRAGBase._Timer":
            self._start = time.perf_counter()
            return self

        def __exit__(self, *_: Any) -> None:
            self.elapsed = time.perf_counter() - self._start

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"k={self.k}, "
            f"relevance_threshold={self.relevance_threshold}, "
            f"vector_store={self.vector_store!r}, "
            f"llm={self.llm!r})"
        )


# ---------------------------------------------------------------------------
# Minimal concrete example subclass — for illustration and unit-testing
# ---------------------------------------------------------------------------

class SimpleAgenticRAG(AgenticRAGBase):
    """
    Minimal concrete implementation of AgenticRAGBase.

    Implements a two-phase agentic loop:
      Phase 1 — Initial retrieve + generate.
      Phase 2 — If ``_should_retrieve_more`` returns ``True``, perform one
                 additional broader retrieval (2× k) with the same query and
                 regenerate the answer.

    This serves as:
      * A runnable reference implementation for testing the base class.
      * A starting template for more complex agentic variants (corrective RAG,
        ReAct, MCTS-based planning, etc.).

    Parameters
    ----------
    vector_store : FAISSVectorStore
        Vector store for retrieval.
    llm : LocalLLM
        Language model for generation and evaluation.
    k : int, optional
        Initial retrieval cardinality (default: 3).
    relevance_threshold : float, optional
        Relevance filter threshold (default: 0.4).
    max_iterations : int, optional
        Maximum number of retrieve-evaluate-generate iterations (default: 2).
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        llm: LocalLLM,
        k: int = 3,
        relevance_threshold: float = 0.4,
        max_iterations: int = 2,
    ) -> None:
        super().__init__(
            vector_store=vector_store,
            llm=llm,
            k=k,
            relevance_threshold=relevance_threshold,
        )
        self.max_iterations = max_iterations

    def query(self, question: str) -> dict[str, Any]:
        """
        Execute a simple two-phase agentic RAG loop.

        Phase 1: Retrieve top-k, generate answer, evaluate.
        Phase 2: (if needed) Retrieve 2×k, regenerate answer.

        Parameters
        ----------
        question : str
            The user's natural-language question.

        Returns
        -------
        dict
            Keys:

            ``answer`` : str
                Final generated answer.
            ``retrieved_docs`` : list[dict]
                All documents used in the final generation.
            ``iterations`` : int
                Number of retrieve-generate iterations performed.
            ``sufficient`` : bool
                Whether the final answer was deemed sufficient.
            ``latency`` : dict
                ``{"iterations": list[float], "total": float}``
        """
        if not question.strip():
            raise ValueError("question must not be an empty string.")

        pipeline_start = time.perf_counter()
        iteration_latencies: list[float] = []
        answer = ""
        docs: list[dict] = []

        for iteration in range(1, self.max_iterations + 1):
            iter_start = time.perf_counter()
            current_k = self.k * iteration  # broaden search on subsequent iterations
            logger.info(
                "%s iteration %d/%d: k=%d",
                self.__class__.__name__, iteration, self.max_iterations, current_k,
            )

            docs = self._retrieve(query=question, k=current_k, filter_by_relevance=True)
            context = self._format_docs(docs, include_scores=True)
            prompt = self._build_answer_prompt(context, question)
            answer = self._generate(prompt)

            iteration_latencies.append(time.perf_counter() - iter_start)
            logger.info(
                "Iteration %d complete (%.4f s). Answer: '%.80s …'",
                iteration, iteration_latencies[-1], answer,
            )

            sufficient = not self._should_retrieve_more(answer, question)
            if sufficient:
                logger.info("Answer deemed SUFFICIENT after iteration %d.", iteration)
                break
        else:
            sufficient = False
            logger.warning(
                "Max iterations (%d) reached without a sufficient answer.",
                self.max_iterations,
            )

        total_latency = time.perf_counter() - pipeline_start

        return {
            "answer": answer,
            "retrieved_docs": docs,
            "iterations": len(iteration_latencies),
            "sufficient": sufficient,
            "latency": {
                "iterations": iteration_latencies,
                "total": total_latency,
            },
        }
