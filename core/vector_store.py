"""
core/vector_store.py
====================
FAISS-backed vector store for dense similarity retrieval in the Agentic RAG pipeline.

Documents are embedded with the provided :class:`Embedder`, normalised to unit
length, and indexed in a ``faiss.IndexFlatIP`` (inner-product index).  Because
vectors are L2-normalised before insertion the inner-product score is equivalent
to the cosine-similarity score.

Dependencies:
    - faiss-cpu  (or faiss-gpu)
    - numpy
    - pickle (stdlib)
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from typing import Any, Optional

import faiss  # type: ignore
import numpy as np

from core.embeddings import Embedder

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    A persistent FAISS vector store that supports adding documents, searching
    by semantic similarity, and saving / loading state to / from disk.

    The store maintains two parallel data structures:

    * ``self._index`` — a ``faiss.IndexFlatIP`` that holds the L2-normalised
      float32 embeddings.
    * ``self._documents`` — a Python list of document dicts, aligned by integer
      position with ``self._index``.  Each entry is::

          {
              'id':       str,
              'text':     str,
              'metadata': dict,
          }

    Cosine similarity is computed via inner-product on L2-normalised vectors,
    which is numerically equivalent and faster than ``IndexFlatL2`` for this
    purpose.

    Args:
        embedder (Embedder): An initialised :class:`~core.embeddings.Embedder`
            instance used to convert text to vectors.
        dimension (int): The dimensionality of the embedding space.
            Must match the output dimension of ``embedder``.
            Defaults to ``384`` (``all-MiniLM-L6-v2``).

    Example:
        >>> from core.embeddings import Embedder
        >>> from core.vector_store import FAISSVectorStore
        >>> embedder = Embedder()
        >>> store = FAISSVectorStore(embedder)
        >>> store.add_documents([{'id': 'doc_1', 'text': 'hello world', 'metadata': {}}])
        >>> results = store.similarity_search("hello", k=1)
        >>> results[0]['id']
        'doc_1'
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, embedder: Embedder, dimension: int = 384) -> None:
        """
        Initialise the vector store.

        Args:
            embedder (Embedder): Embedding model wrapper.
            dimension (int): Embedding dimensionality.  Defaults to 384.
        """
        self.embedder: Embedder = embedder
        self.dimension: int = dimension

        # FAISS inner-product index (cosine sim after L2 norm)
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(dimension)
        # Parallel list of document dicts
        self._documents: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        """
        Embed and add a batch of documents to the index.

        Each document **must** contain the keys ``'id'``, ``'text'``, and
        ``'metadata'``.  The ``'text'`` field is used to produce the embedding;
        all other fields are stored verbatim for retrieval.

        Args:
            documents (list[dict]): List of document dicts, each with::

                {
                    'id':       str,           # unique identifier
                    'text':     str,           # content used for embedding
                    'metadata': dict,          # arbitrary metadata
                }

        Raises:
            ValueError: If any document is missing a required key.
            ValueError: If ``documents`` is empty.

        Example:
            >>> store.add_documents([
            ...     {'id': 'doc_1', 'text': 'Type 2 diabetes is ...', 'metadata': {'domain': 'health'}},
            ... ])
        """
        if not documents:
            raise ValueError("`documents` must be a non-empty list.")

        required_keys = {"id", "text", "metadata"}
        for i, doc in enumerate(documents):
            missing = required_keys - doc.keys()
            if missing:
                raise ValueError(
                    f"Document at index {i} is missing required keys: {missing}"
                )

        texts = [doc["text"] for doc in documents]
        logger.info("Embedding %d document(s) for indexing …", len(texts))
        embeddings: np.ndarray = self.embedder.embed(texts, normalize=True)

        # FAISS expects contiguous float32 arrays
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._index.add(embeddings)
        self._documents.extend(documents)

        logger.info(
            "Index now contains %d document(s).", self._index.ntotal
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def similarity_search(
        self, query: str, k: int = 5
    ) -> list[dict[str, Any]]:
        """
        Retrieve the ``k`` most similar documents to ``query``.

        Similarity is measured as cosine similarity (via inner-product on
        L2-normalised vectors).

        Args:
            query (str): The natural-language query string.
            k (int): Number of results to return.  Defaults to 5.
                Clamped to the number of indexed documents.

        Returns:
            list[dict]: A list of result dicts, each containing all original
            document fields plus a ``'score'`` key (cosine similarity in
            ``[-1, 1]``, higher is better):

            .. code-block:: python

                [
                    {
                        'id':       'doc_001',
                        'text':     '...',
                        'metadata': {...},
                        'score':    0.87,
                    },
                    ...
                ]

        Raises:
            ValueError: If the index is empty.
            ValueError: If ``k`` < 1.

        Example:
            >>> results = store.similarity_search("blood sugar levels", k=3)
            >>> for r in results:
            ...     print(r['score'], r['id'])
        """
        if self._index.ntotal == 0:
            raise ValueError(
                "The vector store is empty. Add documents before searching."
            )
        if k < 1:
            raise ValueError("`k` must be at least 1.")

        # Clamp k to the number of available documents
        k_actual = min(k, self._index.ntotal)

        query_vec = self.embedder.embed_query(query, normalize=True)
        query_vec = np.ascontiguousarray(
            query_vec.reshape(1, -1), dtype=np.float32
        )

        scores, indices = self._index.search(query_vec, k_actual)
        # scores shape: (1, k_actual), indices shape: (1, k_actual)
        scores = scores[0]
        indices = indices[0]

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores, indices):
            if idx == -1:
                # FAISS returns -1 for unfilled slots
                continue
            doc = dict(self._documents[idx])  # shallow copy
            doc["score"] = float(score)
            results.append(doc)

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Persist the FAISS index and document metadata to disk.

        Two files are created:

        * ``<path>.faiss`` — the binary FAISS index.
        * ``<path>.docs``  — the pickled list of document dicts.

        Args:
            path (str): Base path (without extension) for the saved files.
                The parent directory must already exist.

        Raises:
            OSError: If the files cannot be written.

        Example:
            >>> store.save("checkpoints/my_store")
            # Creates: checkpoints/my_store.faiss
            #          checkpoints/my_store.docs
        """
        index_path = f"{path}.faiss"
        docs_path = f"{path}.docs"

        logger.info("Saving FAISS index to %s …", index_path)
        faiss.write_index(self._index, index_path)

        logger.info("Saving document metadata to %s …", docs_path)
        with open(docs_path, "wb") as fh:
            pickle.dump(self._documents, fh, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("Vector store saved successfully.")

    def load(self, path: str) -> None:
        """
        Load a previously saved FAISS index and document metadata from disk.

        Expects the two files created by :meth:`save`:

        * ``<path>.faiss``
        * ``<path>.docs``

        After calling this method the in-memory state is fully replaced by the
        loaded data.

        Args:
            path (str): Base path (without extension) for the files to load.

        Raises:
            FileNotFoundError: If either of the expected files does not exist.
            OSError: If the files cannot be read.

        Example:
            >>> store.load("checkpoints/my_store")
        """
        index_path = f"{path}.faiss"
        docs_path = f"{path}.docs"

        for p in (index_path, docs_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Expected file not found: {p!r}")

        logger.info("Loading FAISS index from %s …", index_path)
        self._index = faiss.read_index(index_path)

        logger.info("Loading document metadata from %s …", docs_path)
        with open(docs_path, "rb") as fh:
            self._documents = pickle.load(fh)

        logger.info(
            "Vector store loaded: %d document(s).", self._index.ntotal
        )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def num_documents(self) -> int:
        """Return the number of documents currently indexed."""
        return self._index.ntotal

    def __repr__(self) -> str:
        return (
            f"FAISSVectorStore("
            f"dimension={self.dimension}, "
            f"num_documents={self.num_documents})"
        )
