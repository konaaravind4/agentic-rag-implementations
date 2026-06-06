"""
core/graph_store.py
===================
NetworkX-based knowledge graph for the Agentic RAG pipeline.

The :class:`KnowledgeGraph` class stores entities as nodes and typed relations
as directed edges.  It supports multi-hop neighbourhood traversal, shortest-
path finding, and lightweight entity extraction from raw document text (noun-
phrase heuristics — no external NLP library required beyond the stdlib ``re``
module).

Dependencies:
    - networkx
"""

from __future__ import annotations

import logging
import re
from collections import deque
from typing import Any, Optional

import networkx as nx  # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lightweight noun-phrase / entity extraction
# ---------------------------------------------------------------------------

# Titles / determiners to strip from candidate noun phrases
_STOP_WORDS: frozenset = frozenset(
    {
        "a", "an", "the", "this", "that", "these", "those",
        "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might",
        "to", "of", "in", "on", "at", "by", "for", "with",
        "and", "or", "but", "not", "also", "its", "their",
        "can", "as", "from", "it", "they", "he", "she",
        "we", "you", "i", "my", "our", "your",
    }
)

# Pattern: capitalised words (proper nouns) or multi-word capitalised phrases
_ENTITY_PATTERN = re.compile(
    r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)*|[A-Z]{2,})\b"
)

# Pattern: noun-like lowercase phrases (>= 2 chars, not stop words)
_NOUN_PATTERN = re.compile(r"\b([a-z][a-z\-]{2,})\b")

# Verbs/prepositions used to infer relations from sentences
_RELATION_PATTERNS: list = [
    (re.compile(r"\bis\b", re.I), "is"),
    (re.compile(r"\bare\b", re.I), "are"),
    (re.compile(r"\bcauses?\b", re.I), "causes"),
    (re.compile(r"\btreat(?:s|ed|ment)?\b", re.I), "treats"),
    (re.compile(r"\buse[sd]?\b", re.I), "uses"),
    (re.compile(r"\binclude[sd]?\b", re.I), "includes"),
    (re.compile(r"\bcontain[sd]?\b", re.I), "contains"),
    (re.compile(r"\bproduced? by\b", re.I), "produced_by"),
    (re.compile(r"\bmanaged? by\b", re.I), "managed_by"),
    (re.compile(r"\bassociated? with\b", re.I), "associated_with"),
    (re.compile(r"\brelated? to\b", re.I), "related_to"),
    (re.compile(r"\bdefined? as\b", re.I), "defined_as"),
    (re.compile(r"\brequire[sd]?\b", re.I), "requires"),
    (re.compile(r"\benables?\b", re.I), "enables"),
    (re.compile(r"\breplace[sd]?\b", re.I), "replaces"),
]


def _extract_entities_from_sentence(sentence: str) -> list:
    """
    Extract candidate entity strings from a single sentence.

    Uses capitalised-word heuristics to pick up proper nouns and technical
    terms.  Stop words are filtered out, and single-character tokens are
    dropped.

    Args:
        sentence (str): A single sentence of text.

    Returns:
        list[str]: Deduplicated list of entity strings found in the sentence.
    """
    candidates: list = []

    # Capitalised phrases (proper nouns, acronyms)
    for match in _ENTITY_PATTERN.finditer(sentence):
        phrase = match.group(1).strip()
        if len(phrase) > 1 and phrase.lower() not in _STOP_WORDS:
            candidates.append(phrase)

    # Prominent lowercase nouns (long enough, not stop words)
    for match in _NOUN_PATTERN.finditer(sentence):
        word = match.group(1)
        if word not in _STOP_WORDS and len(word) >= 4:
            candidates.append(word)

    # Deduplicate while preserving first-seen order
    seen: set = set()
    unique: list = []
    for c in candidates:
        lc = c.lower()
        if lc not in seen:
            seen.add(lc)
            unique.append(c)

    return unique


def _infer_relation(sentence: str) -> str:
    """
    Infer the most likely relation label from a sentence's verb phrase.

    Args:
        sentence (str): A sentence of text.

    Returns:
        str: A relation label string (e.g. ``'causes'``, ``'treats'``).
        Falls back to ``'related_to'`` if no pattern matches.
    """
    for pattern, label in _RELATION_PATTERNS:
        if pattern.search(sentence):
            return label
    return "related_to"


class KnowledgeGraph:
    """
    A directed, attributed knowledge graph backed by ``networkx.DiGraph``.

    The graph stores:

    * **Nodes** -- entities with optional attribute dictionaries.
    * **Edges** -- typed relations between entities, each carrying a
      ``relation`` label and a numeric ``weight``.

    The class also provides a simple ``build_from_documents`` method that
    performs lightweight entity extraction (no external NLP library required)
    to populate the graph automatically from raw text.

    Attributes:
        _graph (nx.DiGraph): The underlying directed NetworkX graph.

    Example:
        >>> kg = KnowledgeGraph()
        >>> kg.add_entity("Insulin", {"type": "hormone"})
        >>> kg.add_entity("Diabetes", {"type": "disease"})
        >>> kg.add_relation("Insulin", "Diabetes", "treats", weight=0.9)
        >>> kg.get_neighbors("Insulin", max_hops=1)
        [{'entity': 'Diabetes', 'relation': 'treats', 'distance': 1}]
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        """Initialise an empty directed knowledge graph."""
        self._graph: nx.DiGraph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Entity management
    # ------------------------------------------------------------------

    def add_entity(
        self, entity: str, attributes: Optional[dict] = None
    ) -> None:
        """
        Add an entity (node) to the graph.

        If the entity already exists its attributes are updated (merged) with
        the provided ``attributes`` dict.

        Args:
            entity (str): A unique string identifier for the entity.
            attributes (dict, optional): Arbitrary key-value metadata to
                attach to the node.  Defaults to ``None`` (empty attributes).

        Raises:
            ValueError: If ``entity`` is an empty string.

        Example:
            >>> kg.add_entity("Metformin", {"class": "biguanide", "use": "antidiabetic"})
        """
        if not entity or not entity.strip():
            raise ValueError("`entity` must be a non-empty string.")

        attrs = attributes or {}
        if self._graph.has_node(entity):
            # Merge attributes into existing node
            self._graph.nodes[entity].update(attrs)
        else:
            self._graph.add_node(entity, **attrs)
            logger.debug("Added entity: %r", entity)

    # ------------------------------------------------------------------
    # Relation management
    # ------------------------------------------------------------------

    def add_relation(
        self,
        source: str,
        target: str,
        relation: str,
        weight: float = 1.0,
    ) -> None:
        """
        Add a directed typed relation (edge) between two entities.

        Both ``source`` and ``target`` nodes are created automatically if
        they do not already exist in the graph.

        If an edge between ``source`` and ``target`` already exists, the
        existing ``relation`` label and ``weight`` are overwritten.

        Args:
            source (str): The source entity identifier.
            target (str): The target entity identifier.
            relation (str): A label describing the relationship
                (e.g. ``'causes'``, ``'treats'``, ``'requires'``).
            weight (float): Numeric strength of the relation.  Must be in
                ``(0, 1]``.  Defaults to ``1.0``.

        Raises:
            ValueError: If ``source`` or ``target`` are empty strings.
            ValueError: If ``weight`` is not in ``(0, 1]``.

        Example:
            >>> kg.add_relation("Type 2 Diabetes", "Metformin", "treated_by", weight=0.95)
        """
        if not source or not source.strip():
            raise ValueError("`source` must be a non-empty string.")
        if not target or not target.strip():
            raise ValueError("`target` must be a non-empty string.")
        if not (0.0 < weight <= 1.0):
            raise ValueError("`weight` must be in the range (0, 1].")

        # Ensure both nodes exist
        self.add_entity(source)
        self.add_entity(target)

        self._graph.add_edge(source, target, relation=relation, weight=weight)
        logger.debug(
            "Added relation: %r -[%s]-> %r (weight=%.2f)",
            source, relation, target, weight,
        )

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    def get_neighbors(
        self, entity: str, max_hops: int = 2
    ) -> list:
        """
        Return all entities reachable from ``entity`` within ``max_hops`` hops.

        Traversal is performed using a breadth-first search (BFS) over the
        directed graph, following outgoing edges.

        Args:
            entity (str): The starting entity.
            max_hops (int): Maximum graph distance to explore.  Defaults to 2.

        Returns:
            list[dict]: A list of result dicts, one per reachable entity
            (excluding the starting entity itself)::

                [
                    {
                        'entity':   'Metformin',
                        'relation': 'treated_by',
                        'distance': 1,
                    },
                    ...
                ]
            Results are sorted by ascending distance.

        Raises:
            ValueError: If ``entity`` is not present in the graph.
            ValueError: If ``max_hops`` < 1.

        Example:
            >>> kg.get_neighbors("Type 2 Diabetes", max_hops=2)
        """
        if not self._graph.has_node(entity):
            raise ValueError(
                f"Entity {entity!r} not found in the knowledge graph."
            )
        if max_hops < 1:
            raise ValueError("`max_hops` must be at least 1.")

        visited: dict = {entity: 0}  # entity -> distance
        results: list = []
        queue: deque = deque()

        # Seed queue with direct neighbours
        for successor in self._graph.successors(entity):
            edge_data = self._graph.edges[entity, successor]
            relation = edge_data.get("relation", "related_to")
            queue.append((successor, 1, relation))

        while queue:
            current, distance, relation = queue.popleft()

            if current in visited:
                continue
            visited[current] = distance
            results.append(
                {
                    "entity": current,
                    "relation": relation,
                    "distance": distance,
                }
            )

            if distance < max_hops:
                for successor in self._graph.successors(current):
                    if successor not in visited:
                        edge_data = self._graph.edges[current, successor]
                        next_relation = edge_data.get("relation", "related_to")
                        queue.append((successor, distance + 1, next_relation))

        results.sort(key=lambda x: x["distance"])
        return results

    def find_path(self, source: str, target: str) -> list:
        """
        Find the shortest directed path between two entities.

        Uses Dijkstra's algorithm via NetworkX, treating edge weights as costs
        (inverted from strength to distance: ``cost = 1 / weight``).

        Args:
            source (str): The starting entity.
            target (str): The destination entity.

        Returns:
            list[str]: An ordered list of entity names forming the shortest
            path, including both ``source`` and ``target``.  Returns an empty
            list if no path exists.

        Raises:
            ValueError: If either ``source`` or ``target`` is not in the graph.

        Example:
            >>> kg.find_path("Insulin Resistance", "Type 2 Diabetes")
            ['Insulin Resistance', 'Metabolic Syndrome', 'Type 2 Diabetes']
        """
        for node, label in ((source, "source"), (target, "target")):
            if not self._graph.has_node(node):
                raise ValueError(
                    f"The {label} entity {node!r} is not present in the graph."
                )

        try:
            path = nx.shortest_path(
                self._graph,
                source=source,
                target=target,
                weight=lambda u, v, d: 1.0 / d.get("weight", 1.0),
            )
            return list(path)
        except nx.NetworkXNoPath:
            logger.debug(
                "No directed path from %r to %r.", source, target
            )
            return []

    # ------------------------------------------------------------------
    # Document-driven construction
    # ------------------------------------------------------------------

    def build_from_documents(self, documents: list) -> None:
        """
        Automatically populate the graph from a list of raw text documents.

        For each sentence in each document the method:

        1. Extracts candidate entities (capitalised words / noun phrases).
        2. Infers a relation label from the sentence's verb phrase.
        3. Adds the first entity as ``source``, the second as ``target``, and
           connects them with the inferred relation.

        This is a heuristic approach that requires no external NLP library.
        For production use, replace with a proper NER + RE model.

        Args:
            documents (list[str]): A list of raw text strings.  Each string
                may contain multiple sentences separated by ``'.'``.

        Raises:
            ValueError: If ``documents`` is empty.

        Example:
            >>> kg = KnowledgeGraph()
            >>> kg.build_from_documents([
            ...     "Insulin treats diabetes. Metformin is an antidiabetic drug.",
            ... ])
        """
        if not documents:
            raise ValueError("`documents` must be a non-empty list.")

        total_entities = 0
        total_relations = 0

        for doc in documents:
            # Split into sentences on '.', '!', or '?'
            sentences = re.split(r"[.!?]+", doc)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:
                    continue

                entities = _extract_entities_from_sentence(sentence)
                if len(entities) < 2:
                    continue

                relation = _infer_relation(sentence)

                # Add entities
                for ent in entities:
                    self.add_entity(ent)
                    total_entities += 1

                # Connect the first two prominent entities in the sentence
                source_ent = entities[0]
                for target_ent in entities[1:3]:  # at most 2 relations per sent
                    if source_ent.lower() != target_ent.lower():
                        self.add_relation(
                            source_ent, target_ent, relation, weight=0.8
                        )
                        total_relations += 1

        logger.info(
            "build_from_documents: added ~%d entity mentions and %d relations "
            "from %d document(s).",
            total_entities,
            total_relations,
            len(documents),
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_triples(self) -> list:
        """
        Return all edges as ``(subject, predicate, object)`` triples.

        Returns:
            list[tuple[str, str, str]]: A list of 3-tuples where each element
            is a string.  The predicate is taken from the ``'relation'`` edge
            attribute; if absent it defaults to ``'related_to'``.

        Example:
            >>> triples = kg.to_triples()
            >>> triples[0]
            ('Insulin', 'treats', 'Diabetes')
        """
        triples: list = []
        for source, target, data in self._graph.edges(data=True):
            relation = data.get("relation", "related_to")
            triples.append((source, relation, target))
        return triples

    # ------------------------------------------------------------------
    # Convenience properties / dunders
    # ------------------------------------------------------------------

    @property
    def num_entities(self) -> int:
        """Return the number of entity nodes in the graph."""
        return self._graph.number_of_nodes()

    @property
    def num_relations(self) -> int:
        """Return the number of relation edges in the graph."""
        return self._graph.number_of_edges()

    def __repr__(self) -> str:
        return (
            f"KnowledgeGraph("
            f"entities={self.num_entities}, "
            f"relations={self.num_relations})"
        )
