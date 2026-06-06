"""
core/embeddings.py
==================
SentenceTransformer-based text embedding module for the Agentic RAG pipeline.

This module provides the `Embedder` class, which wraps the `sentence-transformers`
library to produce dense vector representations of text. Embeddings are used
downstream by the FAISS vector store for similarity search.

Dependencies:
    - sentence-transformers
    - numpy
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    """
    A lightweight wrapper around a SentenceTransformer model for producing
    dense text embeddings.

    The underlying model is loaded lazily on first use to avoid unnecessary
    resource consumption at import time.

    Attributes:
        model_name (str): HuggingFace / SentenceTransformers model identifier.
        _model: The loaded SentenceTransformer model (None until first use).

    Example:
        >>> embedder = Embedder()
        >>> vec = embedder.embed_query("What is diabetes?")
        >>> vec.shape
        (384,)
        >>> batch = embedder.embed(["sentence one", "sentence two"])
        >>> batch.shape
        (2, 384)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialise the Embedder.

        Args:
            model_name (str): The SentenceTransformers model to use.
                Defaults to ``'all-MiniLM-L6-v2'`` which produces 384-dimensional
                embeddings and is fast enough for local CPU inference.
        """
        self.model_name: str = model_name
        self._model: Optional[object] = None  # loaded lazily

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """
        Load the SentenceTransformer model into memory if it has not yet been
        loaded.

        This method is idempotent — calling it multiple times is safe.

        Raises:
            ImportError: If ``sentence-transformers`` is not installed.
            OSError: If the model cannot be downloaded or found locally.
        """
        if self._model is not None:
            return  # already loaded

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        logger.info("Loading SentenceTransformer model: %s", self.model_name)
        self._model = SentenceTransformer(self.model_name)
        logger.info("Model loaded successfully.")

    @staticmethod
    def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
        """
        L2-normalize each row of a 2-D array in-place.

        Args:
            vectors (np.ndarray): Shape ``(n, d)`` — batch of raw embeddings.

        Returns:
            np.ndarray: L2-normalised embeddings (same shape as input).
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero for zero-vectors
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """
        Encode a list of texts into dense vector embeddings.

        Args:
            texts (list[str]): A non-empty list of strings to embed.
            normalize (bool): Whether to L2-normalise the output embeddings.
                Defaults to ``True``, which is required for cosine-similarity
                via inner-product (FAISS ``IndexFlatIP``).

        Returns:
            np.ndarray: A float32 array of shape ``(len(texts), embedding_dim)``.

        Raises:
            ValueError: If ``texts`` is empty.
            ImportError: If ``sentence-transformers`` is not installed.

        Example:
            >>> embedder = Embedder()
            >>> vecs = embedder.embed(["hello world", "foo bar"])
            >>> vecs.shape
            (2, 384)
        """
        if not texts:
            raise ValueError("`texts` must be a non-empty list of strings.")

        self._load_model()

        logger.debug("Embedding %d text(s).", len(texts))
        embeddings: np.ndarray = self._model.encode(  # type: ignore[union-attr]
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        if normalize:
            embeddings = self._l2_normalize(embeddings)

        return embeddings

    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Encode a single query string into a dense vector embedding.

        This is a convenience wrapper around :meth:`embed` that accepts a
        scalar string instead of a list and returns a 1-D array.

        Args:
            query (str): The query text to embed.
            normalize (bool): Whether to L2-normalise the output embedding.
                Defaults to ``True``.

        Returns:
            np.ndarray: A float32 array of shape ``(embedding_dim,)``.

        Raises:
            ValueError: If ``query`` is an empty string.
            ImportError: If ``sentence-transformers`` is not installed.

        Example:
            >>> embedder = Embedder()
            >>> vec = embedder.embed_query("What causes type 2 diabetes?")
            >>> vec.shape
            (384,)
        """
        if not query or not query.strip():
            raise ValueError("`query` must be a non-empty string.")

        embeddings = self.embed([query], normalize=normalize)
        return embeddings[0]  # shape: (embedding_dim,)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        loaded = self._model is not None
        return (
            f"Embedder(model_name={self.model_name!r}, loaded={loaded})"
        )
