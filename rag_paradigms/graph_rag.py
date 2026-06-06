"""
graph_rag.py
============
Implements the Graph RAG paradigm as described in §2.3.4 of:
  "Agentic RAG: A Survey" (arXiv:2501.09136)

Graph RAG augments dense vector retrieval with structured knowledge from a
graph store, enabling multi-hop reasoning across entity relationships that
flat vector search cannot capture.

The pipeline proceeds as follows:
  1. Extract named entities from the query.
  2. Retrieve vector-similar document chunks (broad semantic coverage).
  3. Traverse the knowledge graph starting from the extracted entities,
     following edges up to ``max_hops`` hops to collect related facts and
     neighbouring entity descriptions.
  4. Merge vector results and graph-traversal results into a unified context.
  5. Generate the final answer from the enriched context.

This hybrid approach is especially powerful for:
  * Multi-hop questions ("Who founded the company that built GPT-4?")
  * Relationship queries ("How is X connected to Y?")
  * Structured-domain QA (medical, legal, scientific knowledge bases)
"""

from __future__ import annotations

import re
import time
import logging
from typing import Any

from core.embeddings import Embedder
from core.vector_store import FAISSVectorStore
from core.llm import LocalLLM
from core.graph_store import KnowledgeGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_ENTITY_EXTRACT_PROMPT = (
    "Extract all named entities (people, organisations, places, technologies, "
    "concepts) from the following question. "
    "Output them as a comma-separated list, nothing else. "
    "If there are no named entities, output 'NONE'.\n\n"
    "Question: {question}\n\nEntities:"
)

_ANSWER_PROMPT = (
    "You have access to both retrieved document passages and structured "
    "knowledge graph facts. Use all of the provided information to answer "
    "the question as accurately as possible.\n\n"
    "=== Document Passages ===\n{vector_context}\n\n"
    "=== Knowledge Graph Facts ===\n{graph_context}\n\n"
    "Question: {question}\n\nAnswer:"
)


class GraphRAG:
    """
    Graph RAG Pipeline (§2.3.4).

    Combines dense vector retrieval with multi-hop knowledge graph traversal
    to produce richer, relationship-aware context for answer generation.

    Parameters
    ----------
    vector_store : FAISSVectorStore
        Pre-initialised FAISS vector store (documents must already be indexed).
    llm : LocalLLM
        Language model for entity extraction and answer generation.
    knowledge_graph : KnowledgeGraph
        Pre-populated knowledge graph providing entity lookup and neighbour
        traversal capabilities.
    k : int, optional
        Number of vector-retrieved documents per query (default: 3).

    Expected KnowledgeGraph interface
    ----------------------------------
    The ``knowledge_graph`` object is expected to expose the following methods:

    ``get_entity(entity_name: str) -> dict | None``
        Return a node dict (with at least ``"description"`` key) or ``None``
        if the entity is absent from the graph.

    ``get_neighbours(entity_name: str, hop: int = 1) -> list[dict]``
        Return a list of neighbour dicts at the specified hop distance.
        Each dict should contain ``"entity"``, ``"relation"``, and
        ``"description"`` keys.

    Examples
    --------
    >>> from core.vector_store import FAISSVectorStore
    >>> from core.llm import LocalLLM
    >>> from core.graph_store import KnowledgeGraph
    >>> vs = FAISSVectorStore(dim=768)
    >>> llm = LocalLLM(model_name="mistral-7b")
    >>> kg = KnowledgeGraph(path="data/knowledge_graph.json")
    >>> rag = GraphRAG(vector_store=vs, llm=llm, knowledge_graph=kg, k=3)
    >>> result = rag.query("Who founded OpenAI and what is their mission?")
    >>> print(result["answer"])
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        llm: LocalLLM,
        knowledge_graph: KnowledgeGraph,
        k: int = 3,
    ) -> None:
        """
        Initialise the Graph RAG pipeline.

        Parameters
        ----------
        vector_store : FAISSVectorStore
            Vector store for semantic document retrieval.
        llm : LocalLLM
            Language model for entity extraction and answer generation.
        knowledge_graph : KnowledgeGraph
            Knowledge graph for structured multi-hop retrieval.
        k : int
            Number of documents to retrieve from the vector store.
        """
        self.vector_store = vector_store
        self.llm = llm
        self.knowledge_graph = knowledge_graph
        self.k = k
        self.embedder: Embedder = Embedder()
        logger.info("GraphRAG initialised with k=%d", k)

    # ------------------------------------------------------------------
    # Entity extraction
    # ------------------------------------------------------------------

    def _extract_entities(self, query: str) -> list[str]:
        """
        Extract named entities from the query string.

        Uses an LLM to identify named entities (people, organisations,
        places, technologies, concepts) present in the query.  Falls back
        to simple capitalised-word heuristics if the LLM returns ``"NONE"``
        or an empty response.

        Parameters
        ----------
        query : str
            The user's natural-language question.

        Returns
        -------
        list[str]
            A deduplicated list of entity strings extracted from the query.
            Returns an empty list if no entities can be identified.

        Notes
        -----
        The LLM is asked to output a comma-separated list.  Each item is
        stripped of whitespace and short tokens (< 2 chars) are discarded.
        """
        logger.debug("Extracting entities from query: %.80s", query)
        prompt = _ENTITY_EXTRACT_PROMPT.format(question=query)
        raw_output = self.llm.generate(prompt).strip()

        if not raw_output or raw_output.upper() == "NONE":
            # Fallback: extract capitalised words / noun phrases heuristically
            entities = self._heuristic_entity_extract(query)
            logger.debug("LLM returned NONE; heuristic entities: %s", entities)
            return entities

        # Parse comma-separated list
        entities: list[str] = []
        seen: set[str] = set()
        for token in raw_output.split(","):
            entity = token.strip().strip('"').strip("'")
            if len(entity) >= 2 and entity.lower() not in seen:
                entities.append(entity)
                seen.add(entity.lower())

        logger.debug("Extracted entities: %s", entities)
        return entities

    @staticmethod
    def _heuristic_entity_extract(text: str) -> list[str]:
        """
        Simple heuristic fallback for entity extraction.

        Finds sequences of capitalised words (a rough proxy for proper
        nouns / named entities) using a regular expression.

        Parameters
        ----------
        text : str
            Input text to scan for capitalised tokens.

        Returns
        -------
        list[str]
            Unique capitalised word sequences found in the text, excluding
            single-character tokens and common stop words.
        """
        _STOP = {"The", "A", "An", "Is", "Are", "Was", "Were", "What", "Who",
                 "Where", "When", "How", "Which", "Does", "Do", "Did", "Has",
                 "Have", "Had", "Can", "Could", "Would", "Should", "Will"}

        # Match one or more consecutive Title-Case words
        matches = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        seen: set[str] = set()
        entities: list[str] = []
        for m in matches:
            if m not in _STOP and m not in seen:
                entities.append(m)
                seen.add(m)
        return entities

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    def _graph_retrieve(
        self, entities: list[str], max_hops: int = 2
    ) -> list[str]:
        """
        Perform multi-hop traversal of the knowledge graph.

        For each extracted entity, looks up the entity node and then
        follows edges up to ``max_hops`` levels deep, collecting
        relationship descriptions along the way.

        The traversal is breadth-first: all 1-hop neighbours are explored
        before 2-hop neighbours, etc.  Duplicate nodes are deduplicated by
        entity name.

        Parameters
        ----------
        entities : list[str]
            Seed entity names to start the graph traversal from.
        max_hops : int, optional
            Maximum number of hops from each seed entity (default: 2).
            Higher values explore more of the graph but increase latency.

        Returns
        -------
        list[str]
            A list of human-readable fact strings collected from the graph.
            Each string encodes an entity relationship in the form::

                "<entity> --[relation]--> <neighbour>: <description>"

            Returns an empty list if no entities are found in the graph.
        """
        if not entities:
            logger.debug("No entities provided; skipping graph traversal.")
            return []

        graph_facts: list[str] = []
        visited_entities: set[str] = set()

        for seed_entity in entities:
            # Look up the seed entity in the graph
            entity_node = self.knowledge_graph.get_entity(seed_entity)
            if entity_node is None:
                logger.debug("Entity '%s' not found in knowledge graph.", seed_entity)
                continue

            entity_desc = entity_node.get("description", "")
            if entity_desc and seed_entity not in visited_entities:
                graph_facts.append(f"{seed_entity}: {entity_desc}")
                visited_entities.add(seed_entity)

            # BFS hop expansion
            frontier: list[str] = [seed_entity]
            for hop in range(1, max_hops + 1):
                next_frontier: list[str] = []
                for current_entity in frontier:
                    neighbours = self.knowledge_graph.get_neighbours(
                        current_entity, hop=hop
                    )
                    for nbr in neighbours:
                        neighbour_name: str = nbr.get("entity", "")
                        relation: str = nbr.get("relation", "related_to")
                        description: str = nbr.get("description", "")

                        if neighbour_name and neighbour_name not in visited_entities:
                            fact = (
                                f"{current_entity} --[{relation}]--> "
                                f"{neighbour_name}: {description}"
                            )
                            graph_facts.append(fact)
                            visited_entities.add(neighbour_name)
                            next_frontier.append(neighbour_name)

                frontier = next_frontier
                if not frontier:
                    break  # No new nodes to explore

        logger.debug(
            "Graph traversal collected %d facts from %d entities (max_hops=%d)",
            len(graph_facts),
            len(entities),
            max_hops,
        )
        return graph_facts

    # ------------------------------------------------------------------
    # Context builders
    # ------------------------------------------------------------------

    def _build_vector_context(self, docs: list[dict]) -> str:
        """
        Concatenate vector-retrieved documents into a numbered context string.

        Parameters
        ----------
        docs : list[dict]
            Retrieved documents (each must contain a ``"text"`` key).

        Returns
        -------
        str
            Numbered, newline-separated context string.
        """
        if not docs:
            return "No relevant document passages found."
        return "\n\n".join(f"[{i}] {doc['text']}" for i, doc in enumerate(docs, 1))

    def _build_graph_context(self, graph_facts: list[str]) -> str:
        """
        Format knowledge graph facts into a bulleted context string.

        Parameters
        ----------
        graph_facts : list[str]
            Fact strings from ``_graph_retrieve``.

        Returns
        -------
        str
            Bulleted fact list, or a no-facts message if empty.
        """
        if not graph_facts:
            return "No knowledge graph facts found for the extracted entities."
        return "\n".join(f"• {fact}" for fact in graph_facts)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        max_hops: int = 2,
    ) -> dict[str, Any]:
        """
        Execute the Graph RAG pipeline.

        Pipeline steps
        --------------
        1. Extract named entities from the question (LLM + heuristic fallback).
        2. Embed the question and perform vector store search (top-k docs).
        3. Traverse the knowledge graph from the extracted entities (multi-hop).
        4. Build separate vector and graph context strings.
        5. Merge into a combined prompt and generate the final answer.

        Parameters
        ----------
        question : str
            The user's natural-language question.
        max_hops : int, optional
            Maximum graph traversal hops from each seed entity (default: 2).

        Returns
        -------
        dict
            Result dictionary with keys:

            ``answer`` : str
                The LLM-generated answer.
            ``retrieved_docs`` : list[dict]
                Documents returned by the vector store.
            ``graph_facts`` : list[str]
                Fact strings collected from the knowledge graph.
            ``entities`` : list[str]
                Named entities extracted from the question.
            ``vector_context`` : str
                Formatted context from vector retrieval.
            ``graph_context`` : str
                Formatted context from graph traversal.
            ``prompt`` : str
                The full prompt sent to the LLM.
            ``latency`` : dict
                Per-step timing in seconds::

                    {
                        "entity_extract": float,
                        "vector_retrieve": float,
                        "graph_retrieve":  float,
                        "generate":        float,
                        "total":           float,
                    }

        Raises
        ------
        ValueError
            If ``question`` is empty.
        """
        if not question.strip():
            raise ValueError("question must not be an empty string.")

        latency: dict[str, float] = {}
        pipeline_start = time.perf_counter()

        # ---- Step 1: Entity extraction ----------------------------------------
        t0 = time.perf_counter()
        entities = self._extract_entities(question)
        latency["entity_extract"] = time.perf_counter() - t0
        logger.info("Extracted %d entities in %.4f s: %s", len(entities), latency["entity_extract"], entities)

        # ---- Step 2: Vector retrieval -----------------------------------------
        t0 = time.perf_counter()
        query_embedding = self.embedder.embed(question)
        retrieved_docs: list[dict] = self.vector_store.search(
            query_embedding=query_embedding,
            k=self.k,
        )
        latency["vector_retrieve"] = time.perf_counter() - t0
        logger.info("Vector retrieval returned %d docs in %.4f s", len(retrieved_docs), latency["vector_retrieve"])

        # ---- Step 3: Graph traversal ------------------------------------------
        t0 = time.perf_counter()
        graph_facts = self._graph_retrieve(entities, max_hops=max_hops)
        latency["graph_retrieve"] = time.perf_counter() - t0
        logger.info("Graph traversal collected %d facts in %.4f s", len(graph_facts), latency["graph_retrieve"])

        # ---- Step 4: Build contexts -------------------------------------------
        vector_context = self._build_vector_context(retrieved_docs)
        graph_context = self._build_graph_context(graph_facts)

        # ---- Step 5: Generate answer ------------------------------------------
        prompt = _ANSWER_PROMPT.format(
            vector_context=vector_context,
            graph_context=graph_context,
            question=question,
        )
        t0 = time.perf_counter()
        answer: str = self.llm.generate(prompt)
        latency["generate"] = time.perf_counter() - t0

        latency["total"] = time.perf_counter() - pipeline_start
        logger.info(
            "GraphRAG.query completed in %.4f s "
            "(entity_extract=%.4f, vector=%.4f, graph=%.4f, generate=%.4f)",
            latency["total"],
            latency["entity_extract"],
            latency["vector_retrieve"],
            latency["graph_retrieve"],
            latency["generate"],
        )

        return {
            "answer": answer,
            "retrieved_docs": retrieved_docs,
            "graph_facts": graph_facts,
            "entities": entities,
            "vector_context": vector_context,
            "graph_context": graph_context,
            "prompt": prompt,
            "latency": latency,
        }

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"GraphRAG(k={self.k}, "
            f"vector_store={self.vector_store!r}, "
            f"knowledge_graph={self.knowledge_graph!r}, "
            f"llm={self.llm!r})"
        )
