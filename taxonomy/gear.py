"""
taxonomy/gear.py
================
GeAR: Graph-Enhanced Agent for Retrieval-Augmented Generation — §5.6.2
----------------------------------------------------------------------
'Agentic RAG: A Survey' (arXiv:2501.09136)

GeAR layers a knowledge-graph expansion step on top of a lightweight
BM25 base retriever.  After retrieving an initial set of text documents
the agent traverses entity-relation edges to pull in connected documents,
enriching the context for multi-hop questions.

Pipeline
--------
  Query
   │
   ▼
  BM25Retriever.retrieve()        ←── TF-IDF-style keyword scoring
   │ initial_results
   ▼
  GraphExpansionModule.expand()   ←── BFS over KnowledgeGraph
   │ expanded_results
   ▼
  Agent decision: stop or expand more?  (checks multi-hop signal)
   │
   ▼
  LLM synthesis → answer

References
----------
- Jin et al., "GeAR: Graph-Enhanced Agent for Retrieval-Augmented Generation", 2024.
- Survey §5.6.2.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from core.graph_store import KnowledgeGraph
from core.llm import LocalLLM

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Constants / prompts
# ---------------------------------------------------------------------------

# Keywords that suggest multi-hop reasoning is needed
_MULTI_HOP_KEYWORDS = frozenset([
    "why", "how", "compare", "difference", "both", "between",
    "relationship", "connect", "link", "related", "through",
    "via", "caused", "result", "effect", "impact",
])

_ANSWER_PROMPT = """\
You are a knowledgeable assistant. Use the provided context to answer the
question accurately and concisely. If the context does not contain enough
information, say so.

Context:
{context}

Question: {query}

Answer:"""

_MULTI_HOP_PROMPT = """\
Does the following question require multi-hop reasoning across multiple
concepts or entities?  Answer YES or NO only.

Question: {query}
Answer:"""


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    Simple BM25-like keyword retrieval using TF-IDF-inspired scoring.

    Implements the BM25 Okapi formula:
        BM25(q, d) = Σ IDF(t) * (TF(t, d) * (k1+1)) / (TF(t,d) + k1*(1 - b + b*|d|/avgdl))

    Parameters
    ----------
    k1 : float
        Term saturation parameter (default 1.5).
    b : float
        Length normalisation parameter (default 0.75).

    Example
    -------
    >>> retriever = BM25Retriever()
    >>> retriever.index([{"id": "d1", "text": "Python programming language"}])
    >>> results = retriever.retrieve("Python", k=1)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._corpus: List[Dict[str, Any]] = []
        self._avg_dl: float = 0.0
        self._idf: Dict[str, float] = {}
        self._tokenised: List[List[str]] = []

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, documents: List[Dict[str, Any]]) -> None:
        """
        Build the BM25 index from a list of document dicts.

        Each document must have a ``'text'`` field.  Optional fields
        ``'id'`` and ``'metadata'`` are preserved as-is.

        Parameters
        ----------
        documents : list[dict]
            Documents to index.  Each must have at least ``'text'``.
        """
        self._corpus = list(documents)
        self._tokenised = [
            re.findall(r"\w+", doc["text"].lower()) for doc in documents
        ]
        n_docs = len(documents)
        total_dl = sum(len(tokens) for tokens in self._tokenised)
        self._avg_dl = total_dl / max(n_docs, 1)

        # Compute IDF for each unique term
        df: Dict[str, int] = Counter()
        for tokens in self._tokenised:
            for term in set(tokens):
                df[term] += 1

        self._idf = {
            term: math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in df.items()
        }
        logger.debug("[BM25] Indexed %d documents. avgdl=%.1f", n_docs, self._avg_dl)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k documents by BM25 score.

        Parameters
        ----------
        query : str
            The search query string.
        k : int
            Number of documents to return (default 5).

        Returns
        -------
        list[dict]
            Top-k documents, each containing all original fields plus
            ``'bm25_score'`` and ``'source': 'bm25'``.
        """
        if not self._corpus:
            logger.warning("[BM25] Index is empty. Call index() first.")
            return []

        query_terms = re.findall(r"\w+", query.lower())
        scores: List[Tuple[int, float]] = []

        for idx, doc_tokens in enumerate(self._tokenised):
            dl = len(doc_tokens)
            tf_map = Counter(doc_tokens)
            score = 0.0
            for term in query_terms:
                if term not in self._idf:
                    continue
                tf = tf_map.get(term, 0)
                idf_val = self._idf[term]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * dl / max(self._avg_dl, 1)
                )
                score += idf_val * numerator / (denominator + 1e-9)
            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_k = min(k, len(scores))

        results = []
        for idx, score in scores[:top_k]:
            doc = dict(self._corpus[idx])
            doc["bm25_score"] = round(score, 4)
            doc["score"] = round(score, 4)
            doc.setdefault("source", "bm25")
            results.append(doc)

        return results


# ---------------------------------------------------------------------------
# GraphExpansionModule
# ---------------------------------------------------------------------------

class GraphExpansionModule:
    """
    Expands an initial set of retrieved documents by traversing a
    KnowledgeGraph in BFS order.

    The module:
    1. Extracts entity mentions from each retrieved document's text and the
       query itself.
    2. Locates those entities in the knowledge graph.
    3. BFS-expands up to ``max_depth`` hops to discover connected entities.
    4. Converts discovered entity-relation triples into synthetic text chunks
       and adds them to the result set (de-duplicated).

    Parameters
    ----------
    max_depth : int
        BFS hop limit (default 2).
    max_new_docs : int
        Maximum number of new synthetic chunks to add (default 10).
    """

    def __init__(self, max_depth: int = 2, max_new_docs: int = 10) -> None:
        self.max_depth = max_depth
        self.max_new_docs = max_new_docs

    def _extract_entity_hints(
        self,
        texts: List[str],
        query_entities: List[str],
    ) -> List[str]:
        """
        Collect all candidate entity strings from document texts and query.

        Strategy: take 3+ character alphabetic words that appear in multiple
        texts (suggesting they are important terms), plus any explicit
        ``query_entities``.
        """
        combined_entities = list(query_entities)

        # Frequency across documents to find salient terms
        term_freq: Counter = Counter()
        for text in texts:
            words = re.findall(r"[A-Za-z]{3,}", text)
            for word in set(words):
                term_freq[word.lower()] += 1

        # Add terms that appear in ≥ 2 texts or are capitalised (proper-noun-like)
        for text in texts:
            for word in re.findall(r"[A-Z][a-z]+", text):
                combined_entities.append(word.lower())

        return list(set(combined_entities))

    def expand(
        self,
        initial_results: List[Dict[str, Any]],
        knowledge_graph: KnowledgeGraph,
        query_entities: List[str],
    ) -> List[Dict[str, Any]]:
        """
        BFS graph traversal to find connected documents / entities.

        Parameters
        ----------
        initial_results : list[dict]
            Documents already retrieved by the base retriever.
        knowledge_graph : KnowledgeGraph
            The knowledge graph to traverse.
        query_entities : list[str]
            Entity names extracted from the query.

        Returns
        -------
        list[dict]
            The original *initial_results* plus new graph-derived chunks,
            de-duplicated and sorted by score.
        """
        if knowledge_graph.num_entities == 0:
            logger.debug("[GraphExpansion] Knowledge graph is empty; skipping.")
            return initial_results

        texts = [r.get("text", "") for r in initial_results]
        entity_hints = self._extract_entity_hints(texts, query_entities)

        # Locate seed entities in the graph
        seen_entity_ids: Set[str] = set()
        seed_entities = []
        for hint in entity_hints:
            matched = knowledge_graph.search_entities(hint)
            for ent in matched[:2]:
                if ent.entity_id not in seen_entity_ids:
                    seen_entity_ids.add(ent.entity_id)
                    seed_entities.append(ent)

        if not seed_entities:
            logger.debug("[GraphExpansion] No seed entities matched; returning initial results.")
            return initial_results

        # BFS expansion
        expanded_entities = list(seed_entities)
        frontier = list(seed_entities)

        for _ in range(self.max_depth):
            new_frontier = []
            for entity in frontier:
                neighbours = knowledge_graph.get_neighbors(entity.entity_id)
                for nbr in neighbours:
                    if nbr.entity_id not in seen_entity_ids:
                        seen_entity_ids.add(nbr.entity_id)
                        expanded_entities.append(nbr)
                        new_frontier.append(nbr)
            frontier = new_frontier
            if not frontier:
                break

        # Convert graph entities to text chunks
        new_chunks: List[Dict[str, Any]] = []
        for entity in expanded_entities[:self.max_new_docs]:
            relations = knowledge_graph.get_relations(entity.entity_id)
            rel_parts = []
            for rel in relations[:5]:
                target = knowledge_graph.get_entity(rel.target_id)
                tname = target.name if target else rel.target_id
                rel_parts.append(f"{entity.name} {rel.relation_type} {tname}")

            chunk_text = (
                f"{entity.name} ({entity.entity_type})"
            )
            if rel_parts:
                chunk_text += ": " + "; ".join(rel_parts)

            # Avoid exact duplicates
            if chunk_text not in {r.get("text", "") for r in initial_results}:
                new_chunks.append(
                    {
                        "text":      chunk_text,
                        "source":    "graph_expansion",
                        "entity_id": entity.entity_id,
                        "bm25_score": 0.0,
                        "score":     0.5,  # neutral score for graph-derived chunks
                    }
                )

        combined = initial_results + new_chunks
        logger.info(
            "[GraphExpansion] %d initial + %d new graph chunks = %d total",
            len(initial_results), len(new_chunks), len(combined),
        )
        return combined


# ---------------------------------------------------------------------------
# GeAR
# ---------------------------------------------------------------------------

class GeAR:
    """
    GeAR: Graph-Enhanced Agent for Retrieval-Augmented Generation — §5.6.2.

    Combines BM25 keyword retrieval with knowledge-graph expansion, guided
    by an agent that decides whether multi-hop reasoning is needed.

    Parameters
    ----------
    llm : LocalLLM
        Language model for multi-hop detection, entity extraction, and answer
        synthesis.
    knowledge_graph : KnowledgeGraph
        Populated knowledge graph for graph expansion.
    corpus : list[dict]
        Text corpus to index with BM25 (list of ``{'id': ..., 'text': ...}``).
    top_k : int
        Number of base BM25 results (default 5).
    max_expansion_steps : int
        Maximum graph expansion iterations (default 2).
    """

    def __init__(
        self,
        llm: LocalLLM,
        knowledge_graph: KnowledgeGraph,
        corpus: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 5,
        max_expansion_steps: int = 2,
    ) -> None:
        self.llm = llm
        self.kg = knowledge_graph
        self.top_k = top_k
        self.max_expansion_steps = max_expansion_steps

        self.bm25 = BM25Retriever()
        self.expander = GraphExpansionModule()

        if corpus:
            self.bm25.index(corpus)

    def index(self, corpus: List[Dict[str, Any]]) -> None:
        """(Re-)index the BM25 retriever with a new corpus."""
        self.bm25.index(corpus)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _needs_multi_hop(self, query: str) -> bool:
        """
        Heuristic + optional LLM check for multi-hop reasoning requirement.
        """
        tokens = set(re.findall(r"\w+", query.lower()))
        # Fast heuristic
        if len(tokens & _MULTI_HOP_KEYWORDS) >= 1 or len(query.split()) >= 10:
            return True

        # LLM check for borderline cases
        try:
            prompt = _MULTI_HOP_PROMPT.format(query=query)
            raw = self.llm.generate(prompt, max_new_tokens=8).strip().upper()
            return "YES" in raw
        except Exception:
            return False

    def _extract_query_entities(self, query: str) -> List[str]:
        """Simple regex-based entity extraction from a query."""
        # Capitalised words (excluding first word) + known AI terms
        entities = re.findall(r"(?<!\A)\b[A-Z][a-z]+\b", query)
        # Also grab multi-word proper nouns
        entities += re.findall(r"[A-Z][A-Z]+", query)  # acronyms
        return list(set(e.lower() for e in entities if len(e) > 2))

    @staticmethod
    def _format_context(docs: List[Dict[str, Any]]) -> str:
        """Serialise documents into a context string."""
        if not docs:
            return "No relevant information found."
        parts = []
        for i, doc in enumerate(docs, 1):
            text = doc.get("text", "")
            src  = doc.get("source", "?")
            parts.append(f"[{i}] ({src}) {text}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def query(self, question: str) -> Dict[str, Any]:
        """
        Run the GeAR pipeline for a given question.

        Parameters
        ----------
        question : str
            User's natural-language question.

        Returns
        -------
        dict
            {
              ``base_results``      : list  — initial BM25 results,
              ``expanded_results``  : list  — after graph expansion,
              ``expansion_steps``   : int   — graph expansion iterations,
              ``multi_hop_needed``  : bool  — whether multi-hop was triggered,
              ``answer``            : str   — synthesised answer,
              ``latency``           : float — wall-clock time in seconds,
            }
        """
        logger.info("=== GeAR.query | question=%r ===", question)
        t_start = time.perf_counter()

        # Step 1: BM25 base retrieval
        base_results = self.bm25.retrieve(question, k=self.top_k)
        logger.info("[GeAR] BM25 base retrieval: %d docs", len(base_results))

        # Step 2: Check if multi-hop expansion is needed
        multi_hop = self._needs_multi_hop(question)
        logger.info("[GeAR] Multi-hop needed: %s", multi_hop)

        query_entities = self._extract_query_entities(question)
        current_docs = base_results
        expansion_steps = 0

        if multi_hop:
            for step in range(1, self.max_expansion_steps + 1):
                prev_count = len(current_docs)
                current_docs = self.expander.expand(
                    current_docs, self.kg, query_entities
                )
                expansion_steps = step
                logger.info(
                    "[GeAR] Expansion step %d: %d → %d docs",
                    step, prev_count, len(current_docs),
                )
                # If no new docs were added, stop expanding
                if len(current_docs) == prev_count:
                    break

        # Step 3: Synthesise answer
        context = self._format_context(current_docs)
        prompt = _ANSWER_PROMPT.format(context=context, query=question)
        answer = self.llm.generate(prompt).strip()

        latency = time.perf_counter() - t_start

        return {
            "base_results":     base_results,
            "expanded_results": current_docs,
            "expansion_steps":  expansion_steps,
            "multi_hop_needed": multi_hop,
            "answer":           answer,
            "latency":          round(latency, 4),
        }


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from core.graph_store import build_demo_knowledge_graph

    print("GeAR — smoke test")
    print("=" * 60)

    class _MockLLM:
        model_name = "mock"
        def generate(self, prompt, **_):
            if "multi-hop" in prompt.lower() or "YES" in prompt:
                return "YES"
            return "Mock answer synthesised from retrieved context."

    corpus = [
        {"id": "d1", "text": "RAG combines retrieval with LLM generation."},
        {"id": "d2", "text": "Graph RAG traverses knowledge graphs for multi-hop reasoning."},
        {"id": "d3", "text": "FAISS is used for fast dense vector similarity search."},
        {"id": "d4", "text": "BM25 is a classic keyword-based retrieval algorithm."},
        {"id": "d5", "text": "Transformer models use attention mechanisms for NLP."},
    ]

    kg = build_demo_knowledge_graph()
    gear = GeAR(_MockLLM(), kg, corpus=corpus)

    result = gear.query(
        "How does Graph RAG extend traditional RAG using knowledge graphs?"
    )
    print(f"Base results    : {len(result['base_results'])} docs")
    print(f"Expanded results: {len(result['expanded_results'])} docs")
    print(f"Expansion steps : {result['expansion_steps']}")
    print(f"Multi-hop       : {result['multi_hop_needed']}")
    print(f"Answer          : {result['answer'][:100]}...")
    print(f"Latency         : {result['latency']:.4f}s")
