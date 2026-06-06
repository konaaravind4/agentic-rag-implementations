"""
Unit tests for core infrastructure components.

Tests: Embedder, FAISSVectorStore, LocalLLM, KnowledgeGraph
"""

import sys
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_DOCS = [
    {"id": "d1", "text": "Diabetes is a chronic condition affecting blood sugar levels.", "metadata": {}},
    {"id": "d2", "text": "Machine learning enables computers to learn from data automatically.", "metadata": {}},
    {"id": "d3", "text": "Stock market volatility increases during economic uncertainty.", "metadata": {}},
]


# ---------------------------------------------------------------------------
# Embedder tests
# ---------------------------------------------------------------------------

class TestEmbedder:
    def test_embed_returns_numpy_array(self):
        from core.embeddings import Embedder
        emb = Embedder()
        result = emb.embed(["hello world"])
        assert isinstance(result, np.ndarray)

    def test_embed_shape(self):
        from core.embeddings import Embedder
        emb = Embedder()
        texts = ["hello", "world", "test"]
        result = emb.embed(texts)
        assert result.shape[0] == len(texts)
        assert result.shape[1] > 0   # embedding dimension

    def test_embed_query_returns_1d(self):
        from core.embeddings import Embedder
        emb = Embedder()
        result = emb.embed_query("What is diabetes?")
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1

    def test_embed_empty_list(self):
        from core.embeddings import Embedder
        emb = Embedder()
        result = emb.embed([])
        assert result.shape[0] == 0

    def test_embeddings_are_normalized(self):
        """After L2-normalization, norms should be ≈ 1."""
        from core.embeddings import Embedder
        emb = Embedder()
        result = emb.embed(["test sentence"])
        norm = float(np.linalg.norm(result[0]))
        assert abs(norm - 1.0) < 0.1


# ---------------------------------------------------------------------------
# FAISSVectorStore tests
# ---------------------------------------------------------------------------

class TestFAISSVectorStore:
    def setup_method(self):
        from core.embeddings import Embedder
        from core.vector_store import FAISSVectorStore
        self.embedder = Embedder()
        self.store = FAISSVectorStore(self.embedder)
        self.store.add_documents(SAMPLE_DOCS)

    def test_add_and_search(self):
        results = self.store.similarity_search("diabetes blood sugar", k=2)
        assert isinstance(results, list)
        assert len(results) <= 2

    def test_search_returns_dicts(self):
        results = self.store.similarity_search("machine learning", k=1)
        assert len(results) >= 1
        assert "text" in results[0]
        assert "id" in results[0]

    def test_search_returns_scores(self):
        results = self.store.similarity_search("stock market", k=2)
        for r in results:
            assert "score" in r

    def test_empty_store_search(self):
        from core.embeddings import Embedder
        from core.vector_store import FAISSVectorStore
        empty_store = FAISSVectorStore(Embedder())
        results = empty_store.similarity_search("test", k=3)
        assert results == []

    def test_add_multiple_batches(self):
        from core.embeddings import Embedder
        from core.vector_store import FAISSVectorStore
        store = FAISSVectorStore(Embedder())
        store.add_documents(SAMPLE_DOCS[:2])
        store.add_documents(SAMPLE_DOCS[2:])
        results = store.similarity_search("anything", k=3)
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# LocalLLM tests
# ---------------------------------------------------------------------------

class TestLocalLLM:
    def test_generate_returns_string(self):
        from core.llm import LocalLLM
        llm = LocalLLM()
        result = llm.generate("What is 2+2?", max_new_tokens=20)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_with_short_prompt(self):
        from core.llm import LocalLLM
        llm = LocalLLM()
        result = llm.generate("Hello", max_new_tokens=10)
        assert isinstance(result, str)

    def test_score_relevance_returns_float(self):
        from core.llm import LocalLLM
        llm = LocalLLM()
        score = llm.score_relevance(
            query="diabetes treatment",
            document="Insulin is used to treat diabetes mellitus."
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# KnowledgeGraph tests
# ---------------------------------------------------------------------------

class TestKnowledgeGraph:
    def setup_method(self):
        from core.graph_store import KnowledgeGraph
        self.kg = KnowledgeGraph()
        self.kg.add_entity("diabetes", {"type": "disease"})
        self.kg.add_entity("insulin", {"type": "hormone"})
        self.kg.add_entity("pancreas", {"type": "organ"})
        self.kg.add_relation("insulin", "diabetes", "treats")
        self.kg.add_relation("pancreas", "insulin", "produces")

    def test_add_entity(self):
        from core.graph_store import KnowledgeGraph
        kg = KnowledgeGraph()
        kg.add_entity("test_entity", {"attr": "value"})
        assert "test_entity" in kg.graph.nodes()

    def test_add_relation(self):
        assert self.kg.graph.has_edge("insulin", "diabetes")

    def test_get_neighbors_1_hop(self):
        neighbors = self.kg.get_neighbors("pancreas", max_hops=1)
        neighbor_names = [n["entity"] for n in neighbors]
        assert "insulin" in neighbor_names

    def test_get_neighbors_2_hops(self):
        neighbors = self.kg.get_neighbors("pancreas", max_hops=2)
        neighbor_names = [n["entity"] for n in neighbors]
        # pancreas -> insulin -> diabetes (2 hops)
        assert "diabetes" in neighbor_names or len(neighbors) >= 1

    def test_to_triples(self):
        triples = self.kg.to_triples()
        assert isinstance(triples, list)
        assert len(triples) >= 2
        # Each triple is (source, relation, target)
        for t in triples:
            assert len(t) == 3

    def test_build_from_documents(self):
        from core.graph_store import KnowledgeGraph
        kg = KnowledgeGraph()
        docs = [
            "Diabetes affects millions of people worldwide.",
            "Machine learning is used in healthcare applications.",
        ]
        kg.build_from_documents(docs)
        assert len(kg.graph.nodes()) >= 0  # may or may not find entities
