"""
modular_rag.py
==============
Implements the Modular RAG paradigm as described in §2.3.3 of:
  "Agentic RAG: A Survey" (arXiv:2501.09136)

Modular RAG decomposes the RAG pipeline into discrete, swappable *modules*,
each responsible for exactly one stage of the pipeline.  Modules are
registered as plain Python callables, enabling:

  * Hot-swapping of retrievers, rerankers, or readers at runtime.
  * Composing non-linear pipelines (loops, branches, conditional stages).
  * Independent unit-testing of each stage.
  * A/B testing by replacing a single module without touching the rest.

Pipeline stages (in default order)
-----------------------------------
  ``"pre_retrieve"``   – transform the raw query before embedding.
  ``"retrieve"``       – embed query, perform vector search, return docs.
  ``"rerank"``         – score and reorder retrieved documents.
  ``"build_context"``  – serialise docs into a context string.
  ``"read"``           – generate the final answer from context + question.

Each module callable has a fixed signature::

    module(state: dict) -> dict

where ``state`` is a shared mutable dictionary threaded through the pipeline.
Modules read their inputs from ``state`` and write their outputs back to it.
"""

from __future__ import annotations

import time
import logging
from collections import OrderedDict
from typing import Any, Callable

from core.embeddings import Embedder
from core.vector_store import FAISSVectorStore
from core.llm import LocalLLM

logger = logging.getLogger(__name__)

# Type alias for a module callable
ModuleFn = Callable[[dict], dict]


# ---------------------------------------------------------------------------
# Default module implementations
# ---------------------------------------------------------------------------

def _default_pre_retrieve(state: dict) -> dict:
    """
    Default pre-retrieval module — identity transform.

    Simply copies the original question into ``state["retrieval_query"]``
    without any modification.  Swap this module for ``_rewrite_pre_retrieve``
    or a HyDE module to change pre-retrieval behaviour.

    Parameters
    ----------
    state : dict
        Must contain ``"question"`` (str).

    Returns
    -------
    dict
        ``state`` with ``"retrieval_query"`` set to the original question.
    """
    state["retrieval_query"] = state["question"]
    logger.debug("[pre_retrieve] identity pass-through: '%s'", state["retrieval_query"])
    return state


def _rewrite_pre_retrieve(state: dict) -> dict:
    """
    Optional pre-retrieval module — LLM-based query rewriting.

    Requires ``state["llm"]`` and ``state["question"]``.

    Parameters
    ----------
    state : dict
        Must contain ``"question"`` and ``"llm"``.

    Returns
    -------
    dict
        ``state`` with ``"retrieval_query"`` set to the rewritten query.
    """
    llm: LocalLLM = state["llm"]
    question: str = state["question"]
    prompt = (
        "Rewrite this search query to be more precise and retrieval-friendly. "
        "Output only the rewritten query.\n\nQuery: {q}\n\nRewritten:".format(q=question)
    )
    rewritten = llm.generate(prompt).strip() or question
    state["retrieval_query"] = rewritten
    logger.debug("[pre_retrieve] rewritten query: '%s'", rewritten)
    return state


def _default_retrieve(state: dict) -> dict:
    """
    Default retrieval module — FAISS dense vector search.

    Embeds ``state["retrieval_query"]`` and retrieves the top-k documents
    from the vector store.

    Parameters
    ----------
    state : dict
        Must contain:
        ``"retrieval_query"`` (str), ``"vector_store"`` (FAISSVectorStore),
        ``"embedder"`` (Embedder), ``"k"`` (int).

    Returns
    -------
    dict
        ``state`` with ``"retrieved_docs"`` (list[dict]) populated.
    """
    embedder: Embedder = state["embedder"]
    vector_store: FAISSVectorStore = state["vector_store"]
    k: int = state.get("k", 3)

    query_embedding = embedder.embed(state["retrieval_query"])
    docs = vector_store.search(query_embedding=query_embedding, k=k)
    state["retrieved_docs"] = docs
    logger.debug("[retrieve] returned %d documents", len(docs))
    return state


def _default_rerank(state: dict) -> dict:
    """
    Default reranking module — pass-through (preserves retrieval order).

    This module is intentionally a no-op so the pipeline works without an
    LLM reranker.  Replace it with a cross-encoder or LLM-scoring module
    for improved precision.

    Parameters
    ----------
    state : dict
        Must contain ``"retrieved_docs"`` (list[dict]).

    Returns
    -------
    dict
        ``state`` unchanged (docs remain in retrieval order).
    """
    logger.debug("[rerank] default pass-through; %d docs unchanged", len(state.get("retrieved_docs", [])))
    return state


def _cross_encoder_rerank(state: dict) -> dict:
    """
    Optional reranking module — LLM-based relevance scoring.

    Scores each document against the original question using the LLM and
    reorders in descending relevance.

    Parameters
    ----------
    state : dict
        Must contain ``"question"``, ``"retrieved_docs"``, and ``"llm"``.

    Returns
    -------
    dict
        ``state`` with ``"retrieved_docs"`` sorted by ``"rerank_score"``.
    """
    llm: LocalLLM = state["llm"]
    question: str = state["question"]
    docs: list[dict] = state.get("retrieved_docs", [])

    scored: list[dict] = []
    for doc in docs:
        prompt = (
            "Rate the relevance of this document to the query on a scale 0-10. "
            "Output only the integer.\n\nQuery: {q}\n\nDocument: {d}\n\nScore:".format(
                q=question, d=doc["text"]
            )
        )
        raw = llm.generate(prompt).strip()
        try:
            score = int("".join(filter(str.isdigit, raw.split()[0])))
            score = max(0, min(10, score))
        except (ValueError, IndexError):
            score = 0
        scored.append({**doc, "rerank_score": score})

    state["retrieved_docs"] = sorted(scored, key=lambda d: d["rerank_score"], reverse=True)
    logger.debug("[rerank] cross-encoder reranking complete")
    return state


def _default_build_context(state: dict) -> dict:
    """
    Default context-building module — numbered concatenation.

    Serialises ``state["retrieved_docs"]`` into a single string and stores
    it in ``state["context"]``.

    Parameters
    ----------
    state : dict
        Must contain ``"retrieved_docs"`` (list[dict]).

    Returns
    -------
    dict
        ``state`` with ``"context"`` (str) populated.
    """
    docs: list[dict] = state.get("retrieved_docs", [])
    context = "\n\n".join(f"[{i}] {doc['text']}" for i, doc in enumerate(docs, 1))
    state["context"] = context
    logger.debug("[build_context] context length: %d chars", len(context))
    return state


def _default_read(state: dict) -> dict:
    """
    Default reader module — LLM answer generation.

    Formats the standard answer prompt and calls the LLM.

    Parameters
    ----------
    state : dict
        Must contain ``"context"`` (str), ``"question"`` (str),
        and ``"llm"`` (LocalLLM).

    Returns
    -------
    dict
        ``state`` with ``"answer"`` (str) and ``"prompt"`` (str) populated.
    """
    llm: LocalLLM = state["llm"]
    prompt = (
        "Context: {context}\n\nQuestion: {question}\n\nAnswer:".format(
            context=state.get("context", ""),
            question=state["question"],
        )
    )
    answer = llm.generate(prompt)
    state["answer"] = answer
    state["prompt"] = prompt
    logger.debug("[read] answer generated (first 100 chars): %.100s", answer)
    return state


# ---------------------------------------------------------------------------
# Pipeline stage ordering
# ---------------------------------------------------------------------------

_DEFAULT_STAGE_ORDER: list[str] = [
    "pre_retrieve",
    "retrieve",
    "rerank",
    "build_context",
    "read",
]

_DEFAULT_MODULES: dict[str, ModuleFn] = {
    "pre_retrieve": _default_pre_retrieve,
    "retrieve":     _default_retrieve,
    "rerank":       _default_rerank,
    "build_context": _default_build_context,
    "read":         _default_read,
}

# Public registry of named alternative modules that users can plug in
AVAILABLE_MODULES: dict[str, ModuleFn] = {
    "pre_retrieve.identity":       _default_pre_retrieve,
    "pre_retrieve.rewrite":        _rewrite_pre_retrieve,
    "rerank.passthrough":          _default_rerank,
    "rerank.cross_encoder":        _cross_encoder_rerank,
    "retrieve.faiss":              _default_retrieve,
    "build_context.numbered":      _default_build_context,
    "read.default":                _default_read,
}


# ---------------------------------------------------------------------------
# ModularRAG class
# ---------------------------------------------------------------------------

class ModularRAG:
    """
    Modular RAG Pipeline (§2.3.3).

    Provides a plugin-style architecture where each pipeline stage is an
    independently replaceable callable (module).  Modules communicate via
    a shared ``state`` dictionary that is threaded through the pipeline in
    stage order.

    Parameters
    ----------
    vector_store : FAISSVectorStore
        Pre-initialised vector store for document retrieval.
    llm : LocalLLM
        Language model used by modules that require generation.
    k : int, optional
        Default top-k for the retrieval module (default: 3).

    Examples
    --------
    Basic usage with default modules:

    >>> rag = ModularRAG(vector_store=vs, llm=llm, k=5)
    >>> result = rag.query("What is quantum entanglement?")

    Swapping in LLM-based reranking:

    >>> from rag_paradigms.modular_rag import AVAILABLE_MODULES
    >>> rag.register_module("rerank", AVAILABLE_MODULES["rerank.cross_encoder"])
    >>> result = rag.query("What is quantum entanglement?")

    Adding a custom preprocessing stage:

    >>> def uppercase_query(state):
    ...     state["retrieval_query"] = state["retrieval_query"].upper()
    ...     return state
    >>> rag.register_module("pre_retrieve", uppercase_query)
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        llm: LocalLLM,
        k: int = 3,
    ) -> None:
        """
        Initialise the Modular RAG pipeline with default modules.

        Parameters
        ----------
        vector_store : FAISSVectorStore
            Vector store for document retrieval.
        llm : LocalLLM
            Language model for generation-based modules.
        k : int
            Number of documents to retrieve.
        """
        self.vector_store = vector_store
        self.llm = llm
        self.k = k
        self.embedder: Embedder = Embedder()

        # OrderedDict preserves stage execution order
        self._modules: OrderedDict[str, ModuleFn] = OrderedDict(
            (stage, fn) for stage, fn in _DEFAULT_MODULES.items()
        )
        self._stage_order: list[str] = list(_DEFAULT_STAGE_ORDER)
        logger.info(
            "ModularRAG initialised with stages: %s", self._stage_order
        )

    # ------------------------------------------------------------------
    # Module management
    # ------------------------------------------------------------------

    def register_module(self, stage: str, fn: ModuleFn) -> None:
        """
        Register (or replace) a module for a given pipeline stage.

        Modules must be callables with signature ``(state: dict) -> dict``.
        If ``stage`` is a new name not already in the pipeline, it is
        appended *before* the ``"read"`` stage by default.  Use
        ``set_stage_order()`` to customise stage ordering.

        Parameters
        ----------
        stage : str
            Identifier for the pipeline stage (e.g., ``"rerank"``).
        fn : callable
            Module function with signature ``(state: dict) -> dict``.

        Raises
        ------
        TypeError
            If ``fn`` is not callable.

        Examples
        --------
        >>> rag.register_module("rerank", my_custom_reranker)
        >>> rag.register_module("post_process", strip_hallucinations)
        """
        if not callable(fn):
            raise TypeError(f"fn must be callable, got {type(fn).__name__!r}")

        if stage not in self._modules:
            # Insert new stage before "read" for sensible default ordering
            read_idx = (
                self._stage_order.index("read")
                if "read" in self._stage_order
                else len(self._stage_order)
            )
            self._stage_order.insert(read_idx, stage)
            logger.info("Registered NEW stage '%s' before 'read'", stage)
        else:
            logger.info("Replaced existing module for stage '%s'", stage)

        self._modules[stage] = fn

    def unregister_module(self, stage: str) -> None:
        """
        Remove a module from the pipeline entirely.

        Parameters
        ----------
        stage : str
            Stage identifier to remove.

        Raises
        ------
        KeyError
            If the stage does not exist in the pipeline.
        """
        if stage not in self._modules:
            raise KeyError(f"No module registered for stage '{stage}'")
        del self._modules[stage]
        self._stage_order.remove(stage)
        logger.info("Removed stage '%s' from pipeline", stage)

    def set_stage_order(self, order: list[str]) -> None:
        """
        Redefine the execution order of pipeline stages.

        All stages named in ``order`` must already be registered.

        Parameters
        ----------
        order : list[str]
            Stage names in the desired execution order.

        Raises
        ------
        ValueError
            If any stage in ``order`` is not currently registered.
        """
        unknown = set(order) - set(self._modules)
        if unknown:
            raise ValueError(f"Unknown stages in order: {unknown!r}")
        self._stage_order = list(order)
        logger.info("Pipeline stage order updated to: %s", self._stage_order)

    @property
    def pipeline_summary(self) -> str:
        """Return a human-readable summary of the active pipeline."""
        lines = ["ModularRAG pipeline:"]
        for i, stage in enumerate(self._stage_order, 1):
            fn = self._modules.get(stage)
            fn_name = fn.__name__ if fn else "<missing>"
            lines.append(f"  {i}. [{stage}] → {fn_name}()")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, question: str) -> dict[str, Any]:
        """
        Execute the Modular RAG pipeline for a given question.

        Threads a shared ``state`` dictionary through each registered module
        in ``_stage_order``.  The state is pre-populated with all shared
        resources (LLM, vector store, embedder) so that any module can
        access them without tight coupling to the class.

        Parameters
        ----------
        question : str
            The user's natural-language question.

        Returns
        -------
        dict
            Result dictionary with keys:

            ``answer`` : str
                The generated answer.
            ``retrieved_docs`` : list[dict]
                Documents after the retrieval (and optional reranking) stage.
            ``context`` : str
                Context string fed to the reader module.
            ``retrieval_query`` : str
                The effective query used for vector search.
            ``stages_executed`` : list[str]
                Ordered list of stages that ran successfully.
            ``stage_latencies`` : dict[str, float]
                Per-stage wall-clock time in seconds.
            ``latency`` : dict
                ``{"total": <float>}`` aggregating all stage times.
            ``state`` : dict
                The full pipeline state (useful for debugging).

        Raises
        ------
        ValueError
            If ``question`` is empty.
        RuntimeError
            If a required module is missing for a registered stage.
        """
        if not question.strip():
            raise ValueError("question must not be an empty string.")

        # Initialise shared state ------------------------------------------------
        state: dict[str, Any] = {
            "question": question,
            "llm": self.llm,
            "vector_store": self.vector_store,
            "embedder": self.embedder,
            "k": self.k,
            # Outputs (populated by modules)
            "retrieval_query": "",
            "retrieved_docs": [],
            "context": "",
            "answer": "",
            "prompt": "",
        }

        stage_latencies: dict[str, float] = {}
        stages_executed: list[str] = []
        pipeline_start = time.perf_counter()

        # Run each stage ---------------------------------------------------------
        for stage in self._stage_order:
            module_fn = self._modules.get(stage)
            if module_fn is None:
                raise RuntimeError(
                    f"Stage '{stage}' is in the pipeline order but has no registered module."
                )
            logger.debug("Executing stage: '%s' with module: %s", stage, module_fn.__name__)
            t0 = time.perf_counter()
            try:
                state = module_fn(state)
            except Exception as exc:
                logger.error("Stage '%s' failed with error: %s", stage, exc)
                raise RuntimeError(f"Pipeline stage '{stage}' raised an exception: {exc}") from exc

            stage_latencies[stage] = time.perf_counter() - t0
            stages_executed.append(stage)
            logger.debug("Stage '%s' completed in %.4f s", stage, stage_latencies[stage])

        total_latency = time.perf_counter() - pipeline_start
        logger.info(
            "ModularRAG.query completed in %.4f s | stages: %s",
            total_latency,
            stages_executed,
        )

        return {
            "answer": state.get("answer", ""),
            "retrieved_docs": state.get("retrieved_docs", []),
            "context": state.get("context", ""),
            "retrieval_query": state.get("retrieval_query", question),
            "prompt": state.get("prompt", ""),
            "stages_executed": stages_executed,
            "stage_latencies": stage_latencies,
            "latency": {"total": total_latency},
            "state": state,
        }

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ModularRAG(k={self.k}, stages={self._stage_order}, "
            f"vector_store={self.vector_store!r}, llm={self.llm!r})"
        )


# ---------------------------------------------------------------------------
# Usage examples
# ---------------------------------------------------------------------------

def demo_modularity(vector_store: FAISSVectorStore, llm: LocalLLM) -> None:
    """
    Demonstrate ModularRAG's plug-and-play modularity.

    Illustrates three configurations:
      1. Default pipeline (identity pre-retrieval, pass-through rerank).
      2. Upgraded reranking via ``register_module``.
      3. Upgraded pre-retrieval query rewriting.

    Parameters
    ----------
    vector_store : FAISSVectorStore
        Populated vector store for retrieval.
    llm : LocalLLM
        Language model instance.
    """
    print("=" * 60)
    print("ModularRAG Demo")
    print("=" * 60)

    # --- Config 1: Default ---------------------------------------------------
    rag = ModularRAG(vector_store=vector_store, llm=llm, k=3)
    print("\n[Config 1] Default pipeline")
    print(rag.pipeline_summary)
    result1 = rag.query("What is quantum entanglement?")
    print(f"  Answer (first 80 chars): {result1['answer'][:80]}")
    print(f"  Total latency: {result1['latency']['total']:.4f} s")

    # --- Config 2: Cross-encoder reranking -----------------------------------
    print("\n[Config 2] Swap in LLM-based cross-encoder reranking")
    rag.register_module("rerank", AVAILABLE_MODULES["rerank.cross_encoder"])
    print(rag.pipeline_summary)
    result2 = rag.query("What is quantum entanglement?")
    print(f"  Answer (first 80 chars): {result2['answer'][:80]}")
    print(f"  Total latency: {result2['latency']['total']:.4f} s")

    # --- Config 3: Query rewriting -------------------------------------------
    print("\n[Config 3] Add query rewriting in pre-retrieval")
    rag.register_module("pre_retrieve", AVAILABLE_MODULES["pre_retrieve.rewrite"])
    print(rag.pipeline_summary)
    result3 = rag.query("What is quantum entanglement?")
    print(f"  Rewritten query: {result3['retrieval_query']}")
    print(f"  Answer (first 80 chars): {result3['answer'][:80]}")
    print(f"  Total latency: {result3['latency']['total']:.4f} s")
    print("=" * 60)
