"""
advanced_rag.py
===============
Implements the Advanced RAG paradigm as described in §2.3.2 of:
  "Agentic RAG: A Survey" (arXiv:2501.09136)

Advanced RAG improves upon Naïve RAG by introducing explicit pre- and
post-retrieval optimisation stages:

  Pre-retrieval
  -------------
  * HyDE  – Hypothetical Document Embeddings: generate a plausible
             hypothetical answer first; use its embedding as the retrieval
             query, dramatically improving semantic alignment.
  * Query Rewriting – rephrase the user query via an LLM to be more
             precise and retrieval-friendly.

  Post-retrieval
  --------------
  * Reranking – score each retrieved document for relevance to the query
             and re-order accordingly, filtering noise.
  * Context Compression – ask the LLM to extract only the sentences from
             the concatenated context that are actually relevant to the
             question, reducing irrelevant text in the final prompt.

The ``query()`` method exposes flags to toggle HyDE and reranking so that
any combination of the four enhancements can be benchmarked independently.
"""

from __future__ import annotations

import time
import logging
from typing import Any

from core.embeddings import Embedder
from core.vector_store import FAISSVectorStore
from core.llm import LocalLLM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_HYDE_PROMPT = (
    "Write a short, factual passage that directly answers the following question. "
    "Do not say 'I don't know'. Write only the passage, no preamble.\n\n"
    "Question: {question}\n\nPassage:"
)

_REWRITE_PROMPT = (
    "Rewrite the following search query to be more specific, precise, and "
    "retrieval-friendly. Output only the rewritten query, nothing else.\n\n"
    "Original query: {question}\n\nRewritten query:"
)

_RERANK_PROMPT = (
    "On a scale from 0 to 10, rate how relevant the following document is to "
    "the query. Output only the integer score.\n\n"
    "Query: {query}\n\nDocument:\n{doc_text}\n\nRelevance score (0-10):"
)

_COMPRESS_PROMPT = (
    "Given the following context and question, extract only the sentences from "
    "the context that are directly relevant to answering the question. "
    "Output only the extracted sentences, preserving original wording.\n\n"
    "Question: {question}\n\nContext:\n{context}\n\nRelevant sentences:"
)

_ANSWER_PROMPT = "Context: {context}\n\nQuestion: {question}\n\nAnswer:"


class AdvancedRAG:
    """
    Advanced RAG Pipeline (§2.3.2).

    Extends Naïve RAG with four optional optimisation techniques:
    HyDE, query rewriting (pre-retrieval), reranking, and context
    compression (post-retrieval).

    Parameters
    ----------
    vector_store : FAISSVectorStore
        Pre-initialised FAISS vector store (documents must already be indexed).
    llm : LocalLLM
        Language model used for all generation and scoring tasks.
    k : int, optional
        Number of documents to retrieve from the vector store (default: 5).
        Retrieving more documents before reranking generally improves recall.

    Examples
    --------
    >>> from core.vector_store import FAISSVectorStore
    >>> from core.llm import LocalLLM
    >>> vs = FAISSVectorStore(dim=768)
    >>> llm = LocalLLM(model_name="mistral-7b")
    >>> rag = AdvancedRAG(vector_store=vs, llm=llm, k=5)
    >>> result = rag.query("What is the capital of France?", use_hyde=True, use_rerank=True)
    >>> print(result["answer"])
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        llm: LocalLLM,
        k: int = 5,
    ) -> None:
        """
        Initialise the Advanced RAG pipeline.

        Parameters
        ----------
        vector_store : FAISSVectorStore
            Vector store for similarity search. Documents should be indexed
            prior to calling ``query()``.
        llm : LocalLLM
            Language model for HyDE generation, query rewriting, reranking,
            context compression, and final answer generation.
        k : int
            Number of top documents to retrieve before post-processing.
        """
        self.vector_store = vector_store
        self.llm = llm
        self.k = k
        self.embedder: Embedder = Embedder()
        logger.info("AdvancedRAG initialised with k=%d", k)

    # ------------------------------------------------------------------
    # Pre-retrieval optimisations
    # ------------------------------------------------------------------

    def _hyde(self, query: str) -> str:
        """
        Hypothetical Document Embeddings (HyDE) pre-retrieval step.

        Generates a *hypothetical* (plausible but not verified) passage that
        would answer the query. The embedding of this hypothetical passage is
        then used as the retrieval query instead of the raw question embedding.

        This works because the hypothetical answer occupies a similar latent
        space to real answer documents, reducing the semantic gap between a
        short question and longer document chunks.

        Reference: Gao et al., 2022 – "Precise Zero-Shot Dense Retrieval
        without Relevance Labels" (arXiv:2212.10496).

        Parameters
        ----------
        query : str
            The original user question.

        Returns
        -------
        str
            A short hypothetical passage that plausibly answers the query.
            This string is subsequently embedded and used for vector search.
        """
        logger.debug("HyDE: generating hypothetical document for query: %.80s", query)
        prompt = _HYDE_PROMPT.format(question=query)
        hypothetical_doc = self.llm.generate(prompt).strip()
        logger.debug("HyDE hypothetical doc (first 200 chars): %.200s", hypothetical_doc)
        return hypothetical_doc

    def _rewrite_query(self, query: str) -> str:
        """
        LLM-based query rewriting pre-retrieval step.

        Asks the LLM to rephrase the user's query into a more precise,
        search-engine-friendly form. This handles colloquialisms, ambiguous
        pronouns, and underspecified terms before embedding.

        Parameters
        ----------
        query : str
            The original user question.

        Returns
        -------
        str
            A rewritten version of the query optimised for semantic search.
            Falls back to the original query if the LLM output is empty.
        """
        logger.debug("Query rewrite: rewriting query: %.80s", query)
        prompt = _REWRITE_PROMPT.format(question=query)
        rewritten = self.llm.generate(prompt).strip()
        if not rewritten:
            logger.warning("Query rewrite returned empty string; using original query.")
            return query
        logger.debug("Rewritten query: %s", rewritten)
        return rewritten

    # ------------------------------------------------------------------
    # Post-retrieval optimisations
    # ------------------------------------------------------------------

    def _rerank(self, query: str, docs: list[dict]) -> list[dict]:
        """
        LLM-based reranking post-retrieval step.

        For each retrieved document, asks the LLM to assign a relevance
        score (0–10) relative to the original query. Documents are then
        sorted in descending order of score.

        This filters noise and promotes the most pertinent chunks, which is
        especially valuable when retrieving a large ``k`` with HyDE.

        Parameters
        ----------
        query : str
            The original user question (used for scoring, *not* the HyDE text).
        docs : list[dict]
            Retrieved documents as returned by the vector store, each
            containing at least a ``"text"`` key.

        Returns
        -------
        list[dict]
            The same documents, each augmented with a ``"rerank_score"`` key,
            sorted by ``"rerank_score"`` in descending order.
        """
        logger.debug("Reranking %d documents …", len(docs))
        scored_docs: list[dict] = []

        for doc in docs:
            prompt = _RERANK_PROMPT.format(
                query=query, doc_text=doc["text"]
            )
            raw_score = self.llm.generate(prompt).strip()

            # Safely parse the integer score; default to 0 on failure
            try:
                score = int("".join(filter(str.isdigit, raw_score.split()[0])))
                score = max(0, min(10, score))  # clamp to [0, 10]
            except (ValueError, IndexError):
                logger.warning(
                    "Could not parse rerank score '%s'; defaulting to 0.", raw_score
                )
                score = 0

            scored_docs.append({**doc, "rerank_score": score})

        ranked = sorted(scored_docs, key=lambda d: d["rerank_score"], reverse=True)
        logger.debug(
            "Reranking complete. Top score: %d, bottom score: %d",
            ranked[0]["rerank_score"] if ranked else 0,
            ranked[-1]["rerank_score"] if ranked else 0,
        )
        return ranked

    def _compress_context(self, query: str, context: str) -> str:
        """
        LLM-based context compression post-retrieval step.

        Instructs the LLM to read the full concatenated context and extract
        only the sentences that are directly relevant to the question. This
        reduces the prompt length, lowering hallucination risk and latency.

        Parameters
        ----------
        query : str
            The original user question.
        context : str
            The raw concatenated context built from retrieved documents.

        Returns
        -------
        str
            A compressed context containing only the most relevant sentences.
            Falls back to the original context if compression returns empty.
        """
        logger.debug("Compressing context (original length: %d chars)", len(context))
        prompt = _COMPRESS_PROMPT.format(question=query, context=context)
        compressed = self.llm.generate(prompt).strip()
        if not compressed:
            logger.warning("Context compression returned empty; using original context.")
            return context
        logger.debug(
            "Context compressed from %d → %d chars", len(context), len(compressed)
        )
        return compressed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_context(self, docs: list[dict]) -> str:
        """
        Concatenate retrieved document texts into a numbered context string.

        Parameters
        ----------
        docs : list[dict]
            Documents to concatenate.

        Returns
        -------
        str
            Newline-separated, ordinal-prefixed context string.
        """
        return "\n\n".join(f"[{i}] {doc['text']}" for i, doc in enumerate(docs, 1))

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        use_hyde: bool = True,
        use_rerank: bool = True,
    ) -> dict[str, Any]:
        """
        Execute the Advanced RAG pipeline.

        Pipeline steps
        --------------
        1. (Optional) Query rewriting — always applied when ``use_hyde=False``;
           skipped in favour of HyDE when ``use_hyde=True``.
        2. (Optional) HyDE — generate a hypothetical document, embed it as
           the retrieval query.
        3. Vector store search — retrieve ``k`` candidate documents.
        4. (Optional) Reranking — score and sort documents by relevance.
        5. Context building and compression.
        6. Answer generation.

        Parameters
        ----------
        question : str
            The user's natural-language question.
        use_hyde : bool
            If ``True`` (default), apply HyDE as the pre-retrieval step.
            Query rewriting is used instead when this is ``False``.
        use_rerank : bool
            If ``True`` (default), apply LLM-based reranking after retrieval.

        Returns
        -------
        dict
            Result dictionary with keys:

            ``answer`` : str
                The LLM-generated answer.
            ``retrieved_docs`` : list[dict]
                Documents after optional reranking (each may include a
                ``"rerank_score"`` key).
            ``context`` : str
                The (optionally compressed) context fed to the answer LLM.
            ``rewritten_query`` : str
                The effective retrieval query (HyDE text or rewritten query).
            ``stages`` : dict
                Which optimisation stages were active.
            ``latency`` : dict
                Per-stage timing breakdown in seconds, plus ``"total"``.

        Raises
        ------
        ValueError
            If ``question`` is empty.
        """
        if not question.strip():
            raise ValueError("question must not be an empty string.")

        latency: dict[str, float] = {}
        stages: dict[str, bool] = {
            "hyde": use_hyde,
            "query_rewrite": not use_hyde,
            "rerank": use_rerank,
            "compress": True,  # always applied
        }
        pipeline_start = time.perf_counter()

        # ---- Step 1 & 2: Pre-retrieval ----------------------------------------
        if use_hyde:
            t0 = time.perf_counter()
            retrieval_text = self._hyde(question)
            latency["hyde"] = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            retrieval_text = self._rewrite_query(question)
            latency["query_rewrite"] = time.perf_counter() - t0

        # ---- Step 3: Embed retrieval text + vector search ---------------------
        t0 = time.perf_counter()
        retrieval_embedding = self.embedder.embed(retrieval_text)
        retrieved_docs: list[dict] = self.vector_store.search(
            query_embedding=retrieval_embedding,
            k=self.k,
        )
        latency["retrieve"] = time.perf_counter() - t0
        logger.debug("Retrieved %d docs in %.4f s", len(retrieved_docs), latency["retrieve"])

        # ---- Step 4: Reranking (post-retrieval) --------------------------------
        if use_rerank and retrieved_docs:
            t0 = time.perf_counter()
            retrieved_docs = self._rerank(question, retrieved_docs)
            latency["rerank"] = time.perf_counter() - t0
            logger.debug("Reranking completed in %.4f s", latency["rerank"])

        # ---- Step 5: Build context + compress ----------------------------------
        raw_context = self._build_context(retrieved_docs)

        t0 = time.perf_counter()
        context = self._compress_context(question, raw_context)
        latency["compress"] = time.perf_counter() - t0

        # ---- Step 6: Generate answer -------------------------------------------
        prompt = _ANSWER_PROMPT.format(context=context, question=question)
        t0 = time.perf_counter()
        answer: str = self.llm.generate(prompt)
        latency["generate"] = time.perf_counter() - t0

        latency["total"] = time.perf_counter() - pipeline_start
        logger.info(
            "AdvancedRAG.query completed in %.4f s total (stages=%s)",
            latency["total"],
            stages,
        )

        return {
            "answer": answer,
            "retrieved_docs": retrieved_docs,
            "context": context,
            "raw_context": raw_context,
            "rewritten_query": retrieval_text,
            "prompt": prompt,
            "stages": stages,
            "latency": latency,
        }

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"AdvancedRAG(k={self.k}, "
            f"vector_store={self.vector_store!r}, "
            f"llm={self.llm!r})"
        )
