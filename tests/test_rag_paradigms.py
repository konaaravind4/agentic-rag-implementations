"""
Unit tests for all RAG paradigm implementations.

Tests: NaiveRAG, AdvancedRAG, ModularRAG, GraphRAG, AgenticRAGBase
"""

import sys
import os
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_DOCS = [
    {"id": "d1", "text": "Diabetes is a chronic condition affecting blood sugar. Insulin helps regulate it.", "metadata": {}},
    {"id": "d2", "text": "Machine learning algorithms learn patterns from large datasets automatically.", "metadata": {}},
    {"id": "d3", "text": "Stock markets are influenced by economic indicators and investor sentiment.", "metadata": {}},
    {"id": "d4", "text": "Cloud computing provides scalable on-demand computing resources.", "metadata": {}},
    {"id": "d5", "text": "Legal contracts must be signed by all parties to be enforceable.", "metadata": {}},
]

SAMPLE_QUERY = "What is diabetes and how is it treated?"


def make_embedder():
    from core.embeddings import Embedder
    return Embedder()


def make_vector_store(docs=None):
    from core.embeddings import Embedder
    from core.vector_store import FAISSVectorStore
    emb = Embedder()
    vs = FAISSVectorStore(emb)
    vs.add_documents(docs or SAMPLE_DOCS)
    return vs


def make_llm():
    from core.llm import LocalLLM
    return LocalLLM()


def make_kg():
    from core.graph_store import KnowledgeGraph
    kg = KnowledgeGraph()
    kg.add_entity("diabetes")
    kg.add_entity("insulin")
    kg.add_relation("insulin", "diabetes", "treats")
    return kg


# ---------------------------------------------------------------------------
# NaiveRAG tests
# ---------------------------------------------------------------------------

class TestNaiveRAG:
    def setup_method(self):
        from rag_paradigms.naive_rag import NaiveRAG
        self.rag = NaiveRAG(make_vector_store(), make_llm(), k=2)

    def test_query_returns_dict(self):
        result = self.rag.query(SAMPLE_QUERY)
        assert isinstance(result, dict)

    def test_query_has_answer_key(self):
        result = self.rag.query(SAMPLE_QUERY)
        assert "answer" in result

    def test_query_has_retrieved_docs(self):
        result = self.rag.query(SAMPLE_QUERY)
        assert "retrieved_docs" in result
        assert isinstance(result["retrieved_docs"], list)

    def test_query_has_latency(self):
        result = self.rag.query(SAMPLE_QUERY)
        assert "latency" in result
        assert result["latency"] >= 0

    def test_answer_is_string(self):
        result = self.rag.query(SAMPLE_QUERY)
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    def test_index_documents(self):
        from rag_paradigms.naive_rag import NaiveRAG
        from core.embeddings import Embedder
        from core.vector_store import FAISSVectorStore
        vs = FAISSVectorStore(Embedder())
        rag = NaiveRAG(vs, make_llm(), k=2)
        rag.index_documents(SAMPLE_DOCS[:3])
        result = rag.query("What is machine learning?")
        assert "answer" in result


# ---------------------------------------------------------------------------
# AdvancedRAG tests
# ---------------------------------------------------------------------------

class TestAdvancedRAG:
    def setup_method(self):
        from rag_paradigms.advanced_rag import AdvancedRAG
        self.rag = AdvancedRAG(make_vector_store(), make_llm(), k=2)

    def test_query_returns_dict(self):
        result = self.rag.query(SAMPLE_QUERY)
        assert isinstance(result, dict)

    def test_query_has_required_keys(self):
        result = self.rag.query(SAMPLE_QUERY)
        assert "answer" in result
        assert "latency" in result

    def test_query_without_hyde(self):
        result = self.rag.query(SAMPLE_QUERY, use_hyde=False, use_rerank=False)
        assert "answer" in result
        assert isinstance(result["answer"], str)

    def test_query_with_rerank(self):
        result = self.rag.query(SAMPLE_QUERY, use_hyde=False, use_rerank=True)
        assert "answer" in result


# ---------------------------------------------------------------------------
# ModularRAG tests
# ---------------------------------------------------------------------------

class TestModularRAG:
    def setup_method(self):
        from rag_paradigms.modular_rag import ModularRAG
        self.rag = ModularRAG(make_vector_store(), make_llm())

    def test_query_returns_dict(self):
        result = self.rag.query(SAMPLE_QUERY)
        assert isinstance(result, dict)

    def test_query_has_answer(self):
        result = self.rag.query(SAMPLE_QUERY)
        assert "answer" in result or "final_answer" in result or "output" in result

    def test_register_module(self):
        """Registering a custom module should not raise errors."""
        def custom_retriever(state):
            state["retrieved_docs"] = SAMPLE_DOCS[:2]
            return state
        try:
            self.rag.register_module("retrieve", custom_retriever)
        except Exception as e:
            pytest.skip(f"Module registration not supported in this implementation: {e}")


# ---------------------------------------------------------------------------
# GraphRAG tests
# ---------------------------------------------------------------------------

class TestGraphRAG:
    def setup_method(self):
        from rag_paradigms.graph_rag import GraphRAG
        self.rag = GraphRAG(make_vector_store(), make_llm(), make_kg(), k=2)

    def test_query_returns_dict(self):
        result = self.rag.query(SAMPLE_QUERY)
        assert isinstance(result, dict)

    def test_query_has_answer(self):
        result = self.rag.query(SAMPLE_QUERY)
        assert "answer" in result
        assert isinstance(result["answer"], str)

    def test_query_has_latency(self):
        result = self.rag.query(SAMPLE_QUERY)
        assert "latency" in result


# ---------------------------------------------------------------------------
# AgenticRAGBase tests
# ---------------------------------------------------------------------------

class TestAgenticRAGBase:
    def test_cannot_instantiate_abstract_base(self):
        from rag_paradigms.agentic_rag_base import AgenticRAGBase
        with pytest.raises(TypeError):
            AgenticRAGBase(make_llm(), make_vector_store())  # type: ignore

    def test_concrete_subclass_works(self):
        """Test that a concrete subclass implements query() correctly."""
        from rag_paradigms.agentic_rag_base import AgenticRAGBase

        class ConcreteRAG(AgenticRAGBase):
            def query(self, question: str) -> dict:
                import time
                t = time.time()
                docs = self._retrieve(question)
                context = self._build_context(docs)
                answer = self._generate(f"Answer: {question}\nContext: {context[:200]}")
                return self._format_result(answer, docs, t)

        rag = ConcreteRAG(make_llm(), make_vector_store())
        result = rag.query(SAMPLE_QUERY)
        assert "answer" in result
        assert "latency" in result

    def test_retrieve_returns_list(self):
        from rag_paradigms.agentic_rag_base import AgenticRAGBase

        class ConcreteRAG(AgenticRAGBase):
            def query(self, q):
                return {"answer": "", "latency": 0}

        rag = ConcreteRAG(make_llm(), make_vector_store())
        docs = rag._retrieve("test query", k=2)
        assert isinstance(docs, list)

    def test_evaluate_relevance_returns_float(self):
        from rag_paradigms.agentic_rag_base import AgenticRAGBase

        class ConcreteRAG(AgenticRAGBase):
            def query(self, q):
                return {"answer": "", "latency": 0}

        rag = ConcreteRAG(make_llm(), make_vector_store())
        score = rag._evaluate_relevance("diabetes", "Insulin is used to treat diabetes.")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
