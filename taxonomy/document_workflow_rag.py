"""
taxonomy/document_workflow_rag.py
==================================
Agentic Document Workflows — §5.7
----------------------------------
'Agentic RAG: A Survey' (arXiv:2501.09136)

Implements a full multi-stage document ingestion and agent-driven Q&A
pipeline with citation tracking.

Pipeline stages
---------------
1. **Extract** — Pull raw text and metadata from structured document dicts.
2. **Chunk**   — Split extracted text into overlapping fixed-size chunks.
3. **Index**   — Embed chunks and upsert them into the FAISS vector store.
4. **Query**   — Agent-driven multi-chunk retrieval + answer synthesis with
                 inline ``[doc_id:chunk_id]`` citations.

Citation format
---------------
Every answer includes citations in square brackets referencing the source
document ID and chunk index:  ``[doc_id:chunk_3]``.

References
----------
- Survey §5.7: Agentic document workflows.
- Borgeaud et al., "Improving Language Models by Retrieving from Trillions of Tokens", 2022.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from core.llm import LocalLLM
from core.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYNTHESIS_PROMPT = """\
You are a precise, citation-aware assistant. Answer the following question
using ONLY the provided source chunks. After each factual claim, include a
citation in the format [doc_id:chunk_id].

Source chunks:
{chunks_section}

Question: {query}

Instructions:
- Be concise and factual.
- Add [doc_id:chunk_id] citations after each fact you use.
- If you cannot answer from the context, say "Insufficient information."

Answer:"""


# ---------------------------------------------------------------------------
# DocumentProcessor
# ---------------------------------------------------------------------------

class DocumentProcessor:
    """
    Handles extraction, chunking, and metadata extraction for raw documents.

    Supported document formats
    --------------------------
    A document is any Python dict with at least a ``'text'`` or ``'content'``
    field.  Optional fields: ``'id'``, ``'title'``, ``'metadata'``,
    ``'source'``, ``'author'``, ``'date'``.

    Parameters
    ----------
    chunk_size : int
        Maximum number of words per chunk (default 200).
    chunk_overlap : int
        Number of words to overlap between consecutive chunks (default 30).
    """

    def __init__(
        self,
        chunk_size: int = 200,
        chunk_overlap: int = 30,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("`chunk_overlap` must be smaller than `chunk_size`.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_text(self, document: Dict[str, Any]) -> str:
        """
        Extract the raw text content from a document dict.

        Handles common field names: ``'text'``, ``'content'``, ``'body'``,
        ``'page_content'``, ``'description'``.  Falls back to a string
        representation of the dict if none are found.

        Parameters
        ----------
        document : dict
            A document dict in any of the supported formats.

        Returns
        -------
        str
            The extracted text, stripped of leading/trailing whitespace.

        Raises
        ------
        TypeError
            If ``document`` is not a dict.
        """
        if not isinstance(document, dict):
            raise TypeError(f"Expected dict, got {type(document).__name__}.")

        for field in ("text", "content", "body", "page_content", "description"):
            value = document.get(field)
            if value and isinstance(value, str) and value.strip():
                return value.strip()

        # Last-resort: join all string values
        parts = [str(v) for v in document.values() if isinstance(v, str) and v.strip()]
        return " ".join(parts).strip() or ""

    def chunk_document(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[str]:
        """
        Split text into overlapping word-based chunks.

        Parameters
        ----------
        text : str
            Input text to split.
        chunk_size : int, optional
            Override for ``self.chunk_size``.
        chunk_overlap : int, optional
            Override for ``self.chunk_overlap``.

        Returns
        -------
        list[str]
            List of text chunks.  The final chunk may be shorter than
            ``chunk_size``.  Returns ``['']`` for empty input.
        """
        csize   = chunk_size   if chunk_size   is not None else self.chunk_size
        coverlap = chunk_overlap if chunk_overlap is not None else self.chunk_overlap

        if not text or not text.strip():
            return [""]

        words = text.split()
        if len(words) <= csize:
            return [text.strip()]

        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = min(start + csize, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end == len(words):
                break
            start += csize - coverlap

        return chunks

    def extract_metadata(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured metadata from a document dict.

        Attempts to pull well-known fields and derives additional metadata
        (e.g. a stable document ID from the ``'id'`` field or a hash of text).

        Parameters
        ----------
        document : dict
            Source document dict.

        Returns
        -------
        dict
            Metadata dict with at least: ``doc_id``, ``title``, ``source``,
            ``author``, ``date``, ``domain``, ``word_count``.
        """
        text = self.extract_text(document)
        word_count = len(text.split())

        # Stable doc_id: use explicit field or hash of text
        doc_id = (
            document.get("id")
            or document.get("doc_id")
            or hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
        )

        meta = {
            "doc_id":     str(doc_id),
            "title":      document.get("title", document.get("id", "Untitled")),
            "source":     document.get("source", document.get("url", "unknown")),
            "author":     document.get("author", "unknown"),
            "date":       document.get("date", "unknown"),
            "domain":     document.get("domain",
                          document.get("metadata", {}).get("domain", "general")),
            "word_count": word_count,
        }

        # Pass through any nested metadata fields
        nested = document.get("metadata", {})
        if isinstance(nested, dict):
            for k, v in nested.items():
                meta.setdefault(k, v)

        return meta


# ---------------------------------------------------------------------------
# Chunk record
# ---------------------------------------------------------------------------

class _ChunkRecord:
    """Internal record linking a chunk to its parent document."""

    __slots__ = ("doc_id", "chunk_id", "chunk_index", "text", "metadata")

    def __init__(
        self,
        doc_id: str,
        chunk_index: int,
        text: str,
        metadata: Dict[str, Any],
    ) -> None:
        self.doc_id      = doc_id
        self.chunk_index = chunk_index
        self.chunk_id    = f"chunk_{chunk_index}"
        self.text        = text
        self.metadata    = {**metadata, "chunk_id": self.chunk_id}


# ---------------------------------------------------------------------------
# DocumentWorkflowRAG
# ---------------------------------------------------------------------------

class DocumentWorkflowRAG:
    """
    Multi-stage agentic document workflow with citation-aware Q&A.

    This class implements the full pipeline described in §5.7:

    1. **Document ingestion** — extract, chunk, and index documents.
    2. **Agent-driven Q&A** — retrieve relevant chunks, synthesise an
       answer, and inject ``[doc_id:chunk_id]`` citations.

    Parameters
    ----------
    llm : LocalLLM
        Language model for answer synthesis.
    vector_store : FAISSVectorStore
        FAISS-backed index for chunk storage and retrieval.
    chunk_size : int
        Target chunk size in words (default 200).
    chunk_overlap : int
        Overlap between consecutive chunks in words (default 30).
    top_k : int
        Number of chunks to retrieve per query (default 5).
    """

    def __init__(
        self,
        llm: LocalLLM,
        vector_store: FAISSVectorStore,
        chunk_size: int = 200,
        chunk_overlap: int = 30,
        top_k: int = 5,
    ) -> None:
        self.llm = llm
        self.vector_store = vector_store
        self.top_k = top_k

        self.processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Registry of all indexed chunks (doc_id → list of ChunkRecord)
        self._chunk_registry: Dict[str, List[_ChunkRecord]] = {}
        # Flat lookup: (doc_id, chunk_id) → ChunkRecord
        self._chunk_lookup: Dict[Tuple[str, str], _ChunkRecord] = {}

        logger.info(
            "DocumentWorkflowRAG ready | chunk_size=%d, overlap=%d, top_k=%d",
            chunk_size, chunk_overlap, top_k,
        )

    # ------------------------------------------------------------------
    # Ingestion pipeline
    # ------------------------------------------------------------------

    def process_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Full document ingestion pipeline: extract → chunk → index.

        Parameters
        ----------
        documents : list[dict]
            Raw documents to ingest.  Each must have a text field (``'text'``,
            ``'content'``, etc.) and optionally ``'id'``, ``'metadata'``, etc.

        Returns
        -------
        dict
            {
              ``n_documents``  : int — number of documents processed,
              ``n_chunks``     : int — total chunks created,
              ``doc_ids``      : list[str] — stable document IDs,
              ``latency``      : float — ingestion wall-clock time in seconds,
            }
        """
        logger.info("[Workflow] Processing %d document(s) …", len(documents))
        t_start = time.perf_counter()

        n_chunks_total = 0
        doc_ids = []

        for doc in documents:
            # Stage 1: Extract
            raw_text = self.processor.extract_text(doc)
            meta     = self.processor.extract_metadata(doc)
            doc_id   = meta["doc_id"]
            doc_ids.append(doc_id)

            # Stage 2: Chunk
            chunks = self.processor.chunk_document(raw_text)

            # Stage 3: Build chunk records
            records: List[_ChunkRecord] = []
            for idx, chunk_text in enumerate(chunks):
                if not chunk_text.strip():
                    continue
                record = _ChunkRecord(
                    doc_id=doc_id,
                    chunk_index=idx,
                    text=chunk_text,
                    metadata=meta,
                )
                records.append(record)
                self._chunk_lookup[(doc_id, record.chunk_id)] = record

            self._chunk_registry[doc_id] = records

            # Stage 4: Index into vector store
            texts_to_add = [r.text for r in records]
            metas_to_add = [r.metadata for r in records]
            doc_ids_for_vs = [
                f"{r.doc_id}||{r.chunk_id}" for r in records
            ]

            if texts_to_add:
                try:
                    self.vector_store.add_documents(
                        texts_to_add,
                        metadatas=metas_to_add,
                        doc_ids=doc_ids_for_vs,
                    )
                except TypeError:
                    # Some vector stores use different signatures
                    try:
                        self.vector_store.add_texts(texts_to_add, metadatas=metas_to_add)
                    except Exception as exc:
                        logger.warning("Vector store add failed: %s", exc)

            n_chunks_total += len(records)
            logger.debug(
                "[Workflow] doc_id=%r | %d chunks indexed.", doc_id, len(records)
            )

        latency = time.perf_counter() - t_start
        logger.info(
            "[Workflow] Ingestion done: %d docs, %d chunks in %.3fs",
            len(documents), n_chunks_total, latency,
        )

        return {
            "n_documents": len(documents),
            "n_chunks":    n_chunks_total,
            "doc_ids":     doc_ids,
            "latency":     round(latency, 4),
        }

    # ------------------------------------------------------------------
    # Query pipeline
    # ------------------------------------------------------------------

    def query(self, question: str) -> Dict[str, Any]:
        """
        Agent-driven retrieval and citation-aware answer synthesis.

        Workflow
        --------
        1. Retrieve top-k chunks from the vector store.
        2. Format chunks with their ``[doc_id:chunk_id]`` labels.
        3. Prompt the LLM to generate a grounded, cited answer.
        4. Parse citations from the answer to populate ``citations``.

        Parameters
        ----------
        question : str
            User's natural-language question.

        Returns
        -------
        dict
            {
              ``answer``        : str  — synthesised answer with inline citations,
              ``citations``     : list[str] — unique citation strings found in answer,
              ``source_chunks`` : list[dict] — retrieved chunk details,
              ``latency``       : float — wall-clock time in seconds,
              ``n_docs_retrieved`` : int — number of unique parent docs referenced,
            }
        """
        logger.info("=== DocumentWorkflowRAG.query | question=%r ===", question)
        t_start = time.perf_counter()

        # Step 1: Retrieve
        try:
            raw_results = self.vector_store.search(question, top_k=self.top_k)
        except Exception:
            try:
                raw_results = self.vector_store.similarity_search(
                    question, k=self.top_k
                )
            except Exception as exc:
                logger.error("Retrieval failed: %s", exc)
                raw_results = []

        # Step 2: Format chunks with labels
        chunks_lines: List[str] = []
        source_chunks: List[Dict[str, Any]] = []
        unique_doc_ids: set = set()

        for hit in raw_results:
            meta = hit.get("metadata", {})
            doc_id   = meta.get("doc_id", hit.get("doc_id", "unknown"))
            chunk_id = meta.get("chunk_id", "chunk_0")
            text     = hit.get("text", "")
            score    = hit.get("score", 0.0)

            label = f"[{doc_id}:{chunk_id}]"
            chunks_lines.append(f"{label}\n{text}")
            source_chunks.append(
                {
                    "doc_id":   doc_id,
                    "chunk_id": chunk_id,
                    "text":     text,
                    "score":    score,
                    "label":    label,
                }
            )
            unique_doc_ids.add(doc_id)

        chunks_section = (
            "\n\n".join(chunks_lines)
            if chunks_lines
            else "No relevant chunks found."
        )

        # Step 3: Synthesise answer
        prompt = _SYNTHESIS_PROMPT.format(
            chunks_section=chunks_section, query=question
        )
        answer = self.llm.generate(prompt).strip()

        # Step 4: Parse citations from generated answer
        citation_pattern = re.compile(r"\[[\w_-]+:chunk_\d+\]")
        citations = list(dict.fromkeys(citation_pattern.findall(answer)))

        latency = time.perf_counter() - t_start

        return {
            "answer":             answer,
            "citations":          citations,
            "source_chunks":      source_chunks,
            "latency":            round(latency, 4),
            "n_docs_retrieved":   len(unique_doc_ids),
        }

    # ------------------------------------------------------------------
    # Utility / inspection helpers
    # ------------------------------------------------------------------

    @property
    def n_indexed_chunks(self) -> int:
        """Total number of chunks currently indexed."""
        return sum(len(v) for v in self._chunk_registry.values())

    @property
    def n_indexed_documents(self) -> int:
        """Number of documents whose chunks are indexed."""
        return len(self._chunk_registry)

    def get_chunk(self, doc_id: str, chunk_id: str) -> Optional[_ChunkRecord]:
        """Retrieve a specific chunk record by its IDs."""
        return self._chunk_lookup.get((doc_id, chunk_id))

    def list_documents(self) -> List[Dict[str, str]]:
        """Return a summary list of all indexed documents."""
        summary = []
        for doc_id, records in self._chunk_registry.items():
            if records:
                first = records[0]
                summary.append(
                    {
                        "doc_id":   doc_id,
                        "title":    first.metadata.get("title", doc_id),
                        "n_chunks": str(len(records)),
                    }
                )
        return summary

    def __repr__(self) -> str:
        return (
            f"DocumentWorkflowRAG("
            f"n_docs={self.n_indexed_documents}, "
            f"n_chunks={self.n_indexed_chunks}, "
            f"top_k={self.top_k})"
        )


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("DocumentWorkflowRAG — smoke test")
    print("=" * 60)

    class _MockLLM:
        model_name = "mock"
        def generate(self, prompt, **_):
            return (
                "RAG reduces hallucination by grounding answers in retrieved documents "
                "[ai_001:chunk_0]. The LLM generates conditioned on this context [ai_001:chunk_0]."
            )

    class _MockEmbedder:
        embedding_dim = 4
        def embed(self, texts, **kw):
            import numpy as np
            return np.random.rand(len(texts), 4).astype("float32")
        def embed_query(self, q, **kw):
            import numpy as np
            return np.random.rand(4).astype("float32")

    class _MockVectorStore:
        def __init__(self):
            self._docs = []
        def add_documents(self, texts, metadatas=None, doc_ids=None):
            metas = metadatas or [{}] * len(texts)
            for t, m in zip(texts, metas):
                self._docs.append({"text": t, "metadata": m, "score": 0.9})
        def add_texts(self, texts, metadatas=None):
            self.add_documents(texts, metadatas)
        def search(self, query, top_k=5):
            return self._docs[:top_k]

    from data.sample_documents import SAMPLE_DOCUMENTS

    store = _MockVectorStore()
    rag   = DocumentWorkflowRAG(_MockLLM(), store, chunk_size=50, chunk_overlap=10)

    ingestion_result = rag.process_documents(SAMPLE_DOCUMENTS[:5])
    print(f"Ingested: {ingestion_result['n_documents']} docs, {ingestion_result['n_chunks']} chunks")
    print(f"Indexed docs: {rag.n_indexed_documents}")

    result = rag.query("How does RAG reduce hallucination?")
    print(f"\nAnswer   : {result['answer'][:120]}...")
    print(f"Citations: {result['citations']}")
    print(f"Sources  : {len(result['source_chunks'])} chunks")
    print(f"Latency  : {result['latency']:.4f}s")
