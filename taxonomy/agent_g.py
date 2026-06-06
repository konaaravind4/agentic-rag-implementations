"""
taxonomy/agent_g.py
===================
Agent-G: Agentic Framework for Graph RAG — §5.6.1
--------------------------------------------------
'Agentic RAG: A Survey' (arXiv:2501.09136)

Agent-G introduces a *RetrieverBank* that dynamically selects between a
graph-based and a text-based retriever.  A *CriticModule* evaluates the
quality of retrieved information and triggers a feedback loop when confidence
is low.

Architecture
------------
                        ┌──────────────────┐
    User query ──────►  │  RetrieverBank   │  ◄── selects retriever
                        │  ┌────────────┐  │
                        │  │ GraphAgent │  │  ◄── KnowledgeGraph traversal
                        │  └────────────┘  │
                        │  ┌────────────┐  │
                        │  │ TextAgent  │  │  ◄── FAISS vector search
                        │  └────────────┘  │
                        └────────┬─────────┘
                                 │ retrieved_data
                        ┌────────▼─────────┐
                        │   CriticModule   │  ◄── relevance + confidence scoring
                        └────────┬─────────┘
                      low quality │  high quality
                     ┌───────────┘           └────────────────┐
              re-retrieve with                          LLM synthesis
              refined strategy                         → final answer

References
----------
- He et al., "G-Retriever", 2024.
- Survey §5.6.1: Agent-G.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from core.graph_store import KnowledgeGraph
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

_ENTITY_EXTRACT_PROMPT = """\
Extract the main named entities (people, places, concepts, organisations) from
the following question. List them comma-separated, nothing else.

Question: {query}
Entities:"""

_ANSWER_PROMPT = """\
You are a knowledgeable assistant. Use the provided context to answer the
question accurately and concisely.

Context:
{context}

Question: {query}

Answer:"""

_REFINE_QUERY_PROMPT = """\
The following retrieval attempt produced low-quality results.
Rewrite the query to be more specific and targeted to improve retrieval.

Original query: {query}
Retrieval quality issues: {issues}

Refined query (return ONLY the refined query, nothing else):"""


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _cosine_term_similarity(text_a: str, text_b: str) -> float:
    """Simple TF-cosine similarity between two strings."""
    def _tf(text: str) -> Dict[str, float]:
        tokens = re.findall(r"\w+", text.lower())
        counts = Counter(tokens)
        total = sum(counts.values()) or 1
        return {t: c / total for t, c in counts.items()}

    va, vb = _tf(text_a), _tf(text_b)
    dot = sum(va.get(t, 0.0) * vb.get(t, 0.0) for t in vb)
    mag_a = math.sqrt(sum(v ** 2 for v in va.values())) or 1e-9
    mag_b = math.sqrt(sum(v ** 2 for v in vb.values())) or 1e-9
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# GraphRetrieverAgent
# ---------------------------------------------------------------------------

class GraphRetrieverAgent:
    """
    Retrieves information by traversing a KnowledgeGraph.

    Given a set of query entities, the agent:
    1. Locates matching entity nodes in the graph.
    2. Performs a BFS expansion up to ``max_depth`` hops.
    3. Serialises the discovered subgraph into text chunks.

    Parameters
    ----------
    knowledge_graph : KnowledgeGraph
        The populated knowledge graph to query.
    max_depth : int
        BFS traversal depth (default 2).
    max_entities : int
        Maximum number of entity nodes to expand (default 10).
    """

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        max_depth: int = 2,
        max_entities: int = 10,
    ) -> None:
        self.kg = knowledge_graph
        self.max_depth = max_depth
        self.max_entities = max_entities

    def retrieve(
        self,
        query: str,
        entities: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Traverse the knowledge graph for query-relevant information.

        Parameters
        ----------
        query : str
            Original user query (used for fallback entity extraction).
        entities : list[str], optional
            Pre-extracted entity names.  If omitted, simple keyword matching
            on the query is used.

        Returns
        -------
        list[dict]
            Retrieved chunks, each with ``text``, ``source``, ``entity_id``.
        """
        logger.debug("[GraphAgent] retrieve | entities=%r", entities)

        # Find matching entity nodes
        seed_entities = []
        search_terms = entities or [w for w in query.split() if len(w) > 3]
        for term in search_terms:
            matched = self.kg.search_entities(term)
            seed_entities.extend(matched[:3])

        if not seed_entities:
            logger.debug("[GraphAgent] No seed entities found.")
            return []

        # BFS expansion from seed entities
        seen_ids = set()
        results: List[Dict[str, Any]] = []

        for entity in seed_entities[: self.max_entities]:
            if entity.entity_id in seen_ids:
                continue
            seen_ids.add(entity.entity_id)

            neighbours = self.kg.bfs(entity.entity_id, max_depth=self.max_depth)
            relations = self.kg.get_relations(entity.entity_id)

            # Serialise entity + its direct relations as text
            rel_texts = []
            for rel in relations:
                target = self.kg.get_entity(rel.target_id)
                tname = target.name if target else rel.target_id
                rel_texts.append(
                    f"{entity.name} --[{rel.relation_type}]--> {tname}"
                )

            chunk_text = (
                f"Entity: {entity.name} (type: {entity.entity_type})\n"
                f"Relations: {'; '.join(rel_texts) if rel_texts else 'none'}\n"
                f"Neighbours: {', '.join(n.name for n in neighbours[:5])}"
            )
            results.append(
                {
                    "text": chunk_text,
                    "source": "knowledge_graph",
                    "entity_id": entity.entity_id,
                    "entity_name": entity.name,
                    "score": 1.0,  # graph results treated as maximally relevant seeds
                }
            )

        return results


# ---------------------------------------------------------------------------
# TextRetrieverAgent
# ---------------------------------------------------------------------------

class TextRetrieverAgent:
    """
    Retrieves information via dense vector search in a FAISS store.

    Parameters
    ----------
    vector_store : FAISSVectorStore
        Indexed text corpus.
    top_k : int
        Number of documents to retrieve (default 5).
    """

    def __init__(self, vector_store: FAISSVectorStore, top_k: int = 5) -> None:
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve top-k documents via dense similarity search.

        Parameters
        ----------
        query : str
            User question.

        Returns
        -------
        list[dict]
            Retrieved documents with ``text``, ``score``, ``source``.
        """
        logger.debug("[TextAgent] retrieve | query=%r", query)
        try:
            docs = self.vector_store.search(query, top_k=self.top_k)
        except Exception:
            docs = self.vector_store.similarity_search(query, k=self.top_k)

        for doc in docs:
            doc.setdefault("source", "vector_store")
        return docs


# ---------------------------------------------------------------------------
# RetrieverBank
# ---------------------------------------------------------------------------

class RetrieverBank:
    """
    Dynamically selects and routes to the appropriate retriever.

    Selection strategy
    ------------------
    * If the query contains entity-like tokens (proper nouns, capitalised
      words, or explicit entity hints) **and** the knowledge graph is
      populated → use ``GraphRetrieverAgent`` first, then supplement with
      ``TextRetrieverAgent``.
    * Otherwise → use ``TextRetrieverAgent`` only.

    Parameters
    ----------
    graph_agent : GraphRetrieverAgent
        Graph-based retriever.
    text_agent : TextRetrieverAgent
        Dense-vector-based retriever.
    prefer_graph_threshold : float
        Minimum fraction of query tokens that look like entities before
        preferring the graph retriever (default 0.2 = 20 %).
    """

    def __init__(
        self,
        graph_agent: GraphRetrieverAgent,
        text_agent: TextRetrieverAgent,
        prefer_graph_threshold: float = 0.2,
    ) -> None:
        self.graph_agent = graph_agent
        self.text_agent = text_agent
        self.prefer_graph_threshold = prefer_graph_threshold

    def select_and_retrieve(
        self,
        query: str,
        entities: Optional[List[str]] = None,
        force_both: bool = False,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Select retriever(s) and retrieve documents.

        Parameters
        ----------
        query : str
            User question.
        entities : list[str], optional
            Pre-extracted entity names.
        force_both : bool
            If ``True``, use both retrievers regardless of the selection rule.

        Returns
        -------
        tuple[list[dict], str]
            (retrieved_documents, retrieval_source)
            where ``retrieval_source`` is one of ``'graph'``, ``'text'``,
            or ``'both'``.
        """
        use_graph = force_both or self._should_use_graph(query, entities)
        logger.info(
            "[RetrieverBank] use_graph=%s, force_both=%s", use_graph, force_both
        )

        if force_both:
            graph_docs = self.graph_agent.retrieve(query, entities)
            text_docs  = self.text_agent.retrieve(query)
            combined   = graph_docs + text_docs
            return combined, "both"

        if use_graph:
            graph_docs = self.graph_agent.retrieve(query, entities)
            if graph_docs:
                return graph_docs, "graph"
            # Graph found nothing — fall through to text
            logger.debug("[RetrieverBank] Graph returned nothing; falling back to text.")

        text_docs = self.text_agent.retrieve(query)
        return text_docs, "text"

    # ------------------------------------------------------------------

    def _should_use_graph(
        self, query: str, entities: Optional[List[str]]
    ) -> bool:
        """
        Heuristic: prefer graph retrieval when entity signals are present.
        """
        if entities and len(entities) > 0:
            return True

        tokens = query.split()
        # Count tokens that look like proper nouns (capitalised, not first)
        entity_like = sum(
            1
            for i, tok in enumerate(tokens)
            if i > 0 and tok and tok[0].isupper()
        )
        fraction = entity_like / max(len(tokens), 1)
        return fraction >= self.prefer_graph_threshold


# ---------------------------------------------------------------------------
# CriticModule
# ---------------------------------------------------------------------------

class CriticModule:
    """
    Evaluates the quality of retrieved documents and flags low-confidence
    results that need re-retrieval.

    Scoring
    -------
    Each retrieved document receives:
    * ``relevance_score`` — TF-cosine similarity of document text vs. query.
    * ``coverage_score``  — fraction of query content words covered by the doc.
    * ``combined_score``  — weighted average of the above.

    Parameters
    ----------
    relevance_weight : float
        Weight of TF-cosine relevance in combined score (default 0.7).
    coverage_weight : float
        Weight of coverage score (default 0.3).
    """

    def __init__(
        self,
        relevance_weight: float = 0.7,
        coverage_weight: float = 0.3,
    ) -> None:
        self.relevance_weight = relevance_weight
        self.coverage_weight = coverage_weight

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        query: str,
        retrieved_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Score a list of retrieved documents relative to a query.

        Parameters
        ----------
        query : str
            The original user question.
        retrieved_data : list[dict]
            Documents to score, each expected to have a ``'text'`` field.

        Returns
        -------
        dict
            {
              ``doc_scores``   : list[dict] — per-document score breakdown,
              ``avg_score``    : float      — mean combined score,
              ``max_score``    : float      — best single-document score,
              ``low_confidence`` : bool     — True if avg_score < 0.4,
              ``issues``       : list[str]  — identified quality problems,
            }
        """
        if not retrieved_data:
            return {
                "doc_scores": [],
                "avg_score": 0.0,
                "max_score": 0.0,
                "low_confidence": True,
                "issues": ["No documents retrieved."],
            }

        query_words = set(re.findall(r"\w{3,}", query.lower()))
        scored: List[Dict[str, Any]] = []

        for i, doc in enumerate(retrieved_data):
            text = doc.get("text", "")
            rel_score = _cosine_term_similarity(query, text)
            doc_words = set(re.findall(r"\w{3,}", text.lower()))
            coverage = len(query_words & doc_words) / max(len(query_words), 1)
            combined = (
                self.relevance_weight * rel_score
                + self.coverage_weight * coverage
            )
            scored.append(
                {
                    "doc_index":       i,
                    "relevance_score": round(rel_score, 4),
                    "coverage_score":  round(coverage, 4),
                    "combined_score":  round(combined, 4),
                    "source":          doc.get("source", "unknown"),
                }
            )

        combined_scores = [s["combined_score"] for s in scored]
        avg_score = sum(combined_scores) / len(combined_scores)
        max_score = max(combined_scores)

        issues = []
        if avg_score < 0.2:
            issues.append("Very low average relevance — documents may not address the query.")
        if max_score < 0.3:
            issues.append("No highly relevant document found — consider reformulating the query.")
        if len(retrieved_data) < 2:
            issues.append("Fewer than 2 documents retrieved — coverage may be poor.")

        return {
            "doc_scores":     scored,
            "avg_score":      round(avg_score, 4),
            "max_score":      round(max_score, 4),
            "low_confidence": avg_score < 0.4,
            "issues":         issues,
        }

    def flag_low_confidence(
        self,
        results: List[Dict[str, Any]],
        threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Return only the results whose combined_score is below *threshold*.

        Parameters
        ----------
        results : list[dict]
            Output of :meth:`evaluate`'s ``doc_scores`` list.
        threshold : float
            Score below which a document is considered low-confidence.

        Returns
        -------
        list[dict]
            Subset of *results* with ``combined_score < threshold``.
        """
        return [r for r in results if r.get("combined_score", 1.0) < threshold]


# ---------------------------------------------------------------------------
# AgentG
# ---------------------------------------------------------------------------

class AgentG:
    """
    Agent-G: Agentic Framework for Graph RAG — §5.6.1.

    Orchestrates a dynamic collaboration between graph and text retrievers,
    guided by a critic feedback loop.

    Parameters
    ----------
    llm : LocalLLM
        Language model for entity extraction, query refinement, and answer
        synthesis.
    vector_store : FAISSVectorStore
        Pre-populated vector store for text retrieval.
    knowledge_graph : KnowledgeGraph
        Populated knowledge graph for graph retrieval.
    top_k : int
        Number of documents for text retrieval (default 5).
    max_feedback_iterations : int
        Maximum critic-driven re-retrieval loops (default 2).
    quality_threshold : float
        Critic combined_score below which re-retrieval is triggered (default 0.4).
    """

    def __init__(
        self,
        llm: LocalLLM,
        vector_store: FAISSVectorStore,
        knowledge_graph: KnowledgeGraph,
        top_k: int = 5,
        max_feedback_iterations: int = 2,
        quality_threshold: float = 0.4,
    ) -> None:
        self.llm = llm
        self.vector_store = vector_store
        self.kg = knowledge_graph
        self.top_k = top_k
        self.max_feedback_iterations = max_feedback_iterations
        self.quality_threshold = quality_threshold

        # Sub-components
        self.graph_agent = GraphRetrieverAgent(knowledge_graph)
        self.text_agent  = TextRetrieverAgent(vector_store, top_k=top_k)
        self.retriever_bank = RetrieverBank(self.graph_agent, self.text_agent)
        self.critic = CriticModule()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_entities(self, query: str) -> List[str]:
        """Use the LLM to extract named entities from the query."""
        try:
            prompt = _ENTITY_EXTRACT_PROMPT.format(query=query)
            raw = self.llm.generate(prompt, max_new_tokens=64).strip()
            entities = [e.strip() for e in raw.split(",") if e.strip()]
            return entities[:8]  # cap at 8
        except Exception as exc:
            logger.warning("Entity extraction failed: %s", exc)
            return []

    def _refine_query(self, query: str, issues: List[str]) -> str:
        """Ask the LLM to rewrite the query based on critic issues."""
        try:
            issues_str = "; ".join(issues) if issues else "low relevance"
            prompt = _REFINE_QUERY_PROMPT.format(query=query, issues=issues_str)
            refined = self.llm.generate(prompt, max_new_tokens=64).strip()
            # Make sure we got something sensible
            if refined and len(refined) > 5:
                return refined
        except Exception as exc:
            logger.warning("Query refinement failed: %s", exc)

        # Fallback: append a clarifying phrase
        return f"{query} detailed explanation"

    @staticmethod
    def _format_context(docs: List[Dict[str, Any]]) -> str:
        """Serialise documents into a context string."""
        if not docs:
            return "No relevant information found."
        parts = []
        for i, doc in enumerate(docs, 1):
            text = doc.get("text", "")
            src  = doc.get("source", "unknown")
            parts.append(f"[Source {i} ({src})] {text}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def query(self, question: str) -> Dict[str, Any]:
        """
        Run the Agent-G pipeline for a given question.

        Workflow
        --------
        1. Extract entities from the query.
        2. Select and retrieve via RetrieverBank.
        3. Evaluate quality with CriticModule.
        4. If low quality: refine query, re-retrieve (up to ``max_feedback_iterations``).
        5. Synthesise final answer from best accumulated context.

        Parameters
        ----------
        question : str
            The user's question.

        Returns
        -------
        dict
            {
              ``retrieval_source``    : str   — 'graph', 'text', or 'both',
              ``critic_scores``       : dict  — final critic evaluation,
              ``feedback_iterations`` : int   — re-retrieval cycles triggered,
              ``answer``              : str   — synthesised answer,
              ``latency``             : float — wall-clock time in seconds,
              ``entities_extracted``  : list  — named entities found,
            }
        """
        logger.info("=== AgentG.query | question=%r ===", question)
        t_start = time.perf_counter()

        # Step 1: Entity extraction
        entities = self._extract_entities(question)
        logger.info("[AgentG] Entities extracted: %r", entities)

        # Step 2: Initial retrieval
        all_docs, retrieval_source = self.retriever_bank.select_and_retrieve(
            question, entities=entities
        )
        logger.info(
            "[AgentG] Initial retrieval: %d docs via %s",
            len(all_docs), retrieval_source,
        )

        # Step 3: Critic evaluation + feedback loop
        critic_result = self.critic.evaluate(question, all_docs)
        feedback_iterations = 0
        current_query = question

        while (
            critic_result["low_confidence"]
            and feedback_iterations < self.max_feedback_iterations
        ):
            feedback_iterations += 1
            logger.info(
                "[AgentG] Feedback iteration %d | avg_score=%.3f",
                feedback_iterations, critic_result["avg_score"],
            )

            # Refine query based on issues
            refined_query = self._refine_query(
                current_query, critic_result["issues"]
            )
            current_query = refined_query

            # Re-retrieve with both sources on feedback iterations
            new_docs, new_source = self.retriever_bank.select_and_retrieve(
                refined_query, entities=entities, force_both=True
            )
            all_docs = all_docs + new_docs  # accumulate
            retrieval_source = "both"

            # Re-evaluate
            critic_result = self.critic.evaluate(question, all_docs)

        # Step 4: Synthesise answer
        context = self._format_context(all_docs)
        prompt = _ANSWER_PROMPT.format(context=context, query=question)
        answer = self.llm.generate(prompt).strip()

        latency = time.perf_counter() - t_start

        return {
            "retrieval_source":    retrieval_source,
            "critic_scores":       critic_result,
            "feedback_iterations": feedback_iterations,
            "answer":              answer,
            "latency":             round(latency, 4),
            "entities_extracted":  entities,
            "retrieved_docs":      all_docs,
        }


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from core.graph_store import build_demo_knowledge_graph

    print("AgentG — smoke test")
    print("=" * 60)

    class _MockLLM:
        model_name = "mock"
        def generate(self, prompt, **_):
            if "Entities" in prompt:
                return "RAG, LLM, FAISS"
            if "Refined" in prompt:
                return "detailed explanation of RAG"
            return "This is a mock answer about the query."

    class _MockStore:
        def search(self, query, top_k=5):
            return [{"text": f"Info about {query}", "score": 0.8, "source": "text"}]
        def similarity_search(self, query, k=5):
            return self.search(query, top_k=k)

    kg = build_demo_knowledge_graph()
    rag = AgentG(_MockLLM(), _MockStore(), kg)

    result = rag.query("How does Graph RAG extend traditional RAG?")
    print(f"Retrieval source    : {result['retrieval_source']}")
    print(f"Feedback iterations : {result['feedback_iterations']}")
    print(f"Critic avg score    : {result['critic_scores']['avg_score']:.3f}")
    print(f"Answer              : {result['answer'][:100]}...")
    print(f"Latency             : {result['latency']:.4f}s")
