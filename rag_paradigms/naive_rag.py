"""
naive_rag.py
============
Implements the Naïve RAG paradigm as described in §2.3.1 of:
  "Agentic RAG: A Survey" (arXiv:2501.09136)

Naïve RAG is the simplest form of Retrieval-Augmented Generation:
  1. Index documents into a vector store at ingestion time.
  2. At query time, embed the raw user question.
  3. Retrieve the top-k most similar document chunks.
  4. Concatenate retrieved chunks into a context string.
  5. Pass context + question to an LLM to generate the final answer.

No query rewriting, reranking, or iterative retrieval is performed.
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
# Prompt template
# ---------------------------------------------------------------------------
_ANSWER_PROMPT = "Context: {context}\n\nQuestion: {question}\n\nAnswer:"


class NaiveRAG:
    """
    Naïve RAG Pipeline (§2.3.1).

    The simplest RAG paradigm: a single-pass retrieve-then-read approach
    with no pre- or post-retrieval optimisation.

    Parameters
    ----------
    vector_store : FAISSVectorStore
        Pre-initialised FAISS vector store used for document indexing and
        similarity search.
    llm : LocalLLM
        Language model instance used for answer generation.
    k : int, optional
        Number of documents to retrieve per query (default: 3).

    Examples
    --------
    >>> from core.vector_store import FAISSVectorStore
    >>> from core.llm import LocalLLM
    >>> vs = FAISSVectorStore(dim=768)
    >>> llm = LocalLLM(model_name="mistral-7b")
    >>> rag = NaiveRAG(vector_store=vs, llm=llm, k=5)
    >>> rag.index_documents([{"id": "1", "text": "Python is a programming language."}])
    >>> result = rag.query("What is Python?")
    >>> print(result["answer"])
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        llm: LocalLLM,
        k: int = 3,
    ) -> None:
        """
        Initialise the Naïve RAG pipeline.

        Parameters
        ----------
        vector_store : FAISSVectorStore
            Vector store for document embeddings.
        llm : LocalLLM
            Language model for answer generation.
        k : int
            Top-k documents to retrieve per query.
        """
        self.vector_store = vector_store
        self.llm = llm
        self.k = k
        self.embedder: Embedder = Embedder()
        logger.info("NaiveRAG initialised with k=%d", k)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_documents(self, documents: list[dict]) -> None:
        """
        Embed and index a list of documents into the vector store.

        Each document dictionary must contain at least a ``"text"`` key.
        An optional ``"id"`` key is used as the document identifier; if
        absent, the list index is used instead.

        Parameters
        ----------
        documents : list[dict]
            List of document dicts, e.g.::

                [
                    {"id": "doc-1", "text": "..."},
                    {"id": "doc-2", "text": "...", "metadata": {...}},
                ]

        Raises
        ------
        ValueError
            If ``documents`` is empty or any document lacks a ``"text"`` key.
        """
        if not documents:
            raise ValueError("documents list must not be empty.")

        texts: list[str] = []
        ids: list[str] = []
        metadatas: list[dict] = []

        for i, doc in enumerate(documents):
            if "text" not in doc:
                raise ValueError(f"Document at index {i} is missing the 'text' key.")
            texts.append(doc["text"])
            ids.append(str(doc.get("id", i)))
            metadatas.append({k: v for k, v in doc.items() if k not in ("text", "id")})

        logger.info("Embedding %d documents …", len(texts))
        embeddings = self.embedder.embed_batch(texts)

        self.vector_store.add(
            embeddings=embeddings,
            texts=texts,
            ids=ids,
            metadatas=metadatas,
        )
        logger.info("Indexed %d documents into the vector store.", len(texts))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_context(self, retrieved_docs: list[dict]) -> str:
        """
        Concatenate the text of retrieved documents into a single context string.

        Documents are separated by double newlines and prefixed with an
        ordinal marker for clarity in the prompt.

        Parameters
        ----------
        retrieved_docs : list[dict]
            Each dict must contain a ``"text"`` key (returned by the vector
            store search).

        Returns
        -------
        str
            A single context string ready to be inserted into the prompt.
        """
        parts: list[str] = []
        for i, doc in enumerate(retrieved_docs, start=1):
            parts.append(f"[{i}] {doc['text']}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, question: str) -> dict[str, Any]:
        """
        Execute the Naïve RAG pipeline for a given question.

        Pipeline steps
        --------------
        1. Embed the raw question.
        2. Retrieve the top-k documents from the vector store.
        3. Concatenate the retrieved texts into a context string.
        4. Format the prompt using the standard template.
        5. Generate and return the answer.

        Parameters
        ----------
        question : str
            The user's natural-language question.

        Returns
        -------
        dict
            A result dictionary with the following keys:

            ``answer`` : str
                The LLM-generated answer.
            ``retrieved_docs`` : list[dict]
                The raw documents returned by the vector store, each
                containing ``id``, ``text``, ``score``, and ``metadata``.
            ``context`` : str
                The concatenated context string fed to the LLM.
            ``prompt`` : str
                The full prompt sent to the LLM.
            ``latency`` : dict
                Timing breakdown in seconds::

                    {
                        "embed":    <float>,  # query embedding
                        "retrieve": <float>,  # vector store search
                        "generate": <float>,  # LLM generation
                        "total":    <float>,  # end-to-end wall time
                    }

        Raises
        ------
        ValueError
            If ``question`` is an empty string.
        RuntimeError
            If the vector store contains no indexed documents.
        """
        if not question.strip():
            raise ValueError("question must not be an empty string.")

        latency: dict[str, float] = {}
        pipeline_start = time.perf_counter()

        # ---- Step 1: Embed the query ----------------------------------------
        t0 = time.perf_counter()
        query_embedding = self.embedder.embed(question)
        latency["embed"] = time.perf_counter() - t0
        logger.debug("Query embedded in %.4f s", latency["embed"])

        # ---- Step 2: Retrieve top-k documents --------------------------------
        t0 = time.perf_counter()
        retrieved_docs: list[dict] = self.vector_store.search(
            query_embedding=query_embedding,
            k=self.k,
        )
        latency["retrieve"] = time.perf_counter() - t0
        logger.debug(
            "Retrieved %d docs in %.4f s", len(retrieved_docs), latency["retrieve"]
        )

        # ---- Step 3: Build context -------------------------------------------
        context = self._build_context(retrieved_docs)

        # ---- Step 4: Format prompt -------------------------------------------
        prompt = _ANSWER_PROMPT.format(context=context, question=question)
        logger.debug("Prompt (first 200 chars): %.200s …", prompt)

        # ---- Step 5: Generate answer -----------------------------------------
        t0 = time.perf_counter()
        answer: str = self.llm.generate(prompt)
        latency["generate"] = time.perf_counter() - t0
        logger.debug("Answer generated in %.4f s", latency["generate"])

        latency["total"] = time.perf_counter() - pipeline_start
        logger.info(
            "NaiveRAG.query completed in %.4f s (embed=%.4f, retrieve=%.4f, generate=%.4f)",
            latency["total"],
            latency["embed"],
            latency["retrieve"],
            latency["generate"],
        )

        return {
            "answer": answer,
            "retrieved_docs": retrieved_docs,
            "context": context,
            "prompt": prompt,
            "latency": latency,
        }

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"NaiveRAG(k={self.k}, "
            f"vector_store={self.vector_store!r}, "
            f"llm={self.llm!r})"
        )
