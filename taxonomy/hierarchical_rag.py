"""
taxonomy/hierarchical_rag.py
============================
Hierarchical Agentic RAG (§5.3)
---------------------------------
Architecture from: "Agentic RAG: A Survey" (arXiv:2501.09136)

In Hierarchical Agentic RAG, agents are arranged in a **3-tier command
hierarchy**:

  Tier 1 – Strategic (top)
  ┗━ ``StrategicOrchestratorAgent``
       Examines the query, classifies its domain(s), and decides which
       Tier-2 domain agents to activate.

  Tier 2 – Tactical (mid)
  ┗━ ``MedicalDomainAgent``   (activated for medical/health queries)
  ┗━ ``FinanceDomainAgent``   (activated for finance/economics queries)
  ┗━ ``GeneralDomainAgent``   (catch-all, always available)
       Each domain agent receives the query and decides which Tier-3
       retrieval primitives to invoke.

  Tier 3 – Operational (low)
  ┗━ ``VectorRetrieverAgent``  – FAISS dense search
  ┗━ ``GraphRetrieverAgent``   – Knowledge-graph triple lookup (simulated)
  ┗━ ``KeywordRetrieverAgent`` – BM25-style keyword search

Commands flow downward; results bubble back up through each tier until
the Tier-1 orchestrator composes the final answer.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter
from typing import Any

from core.vector_store import FAISSVectorStore
from core.llm import LocalLLM
from core.graph_store import KnowledgeGraph

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Demo knowledge graph (simulated triples)
# ---------------------------------------------------------------------------
_KG_TRIPLES: list[dict] = [
    {"subject": "machine learning", "predicate": "is_a", "object": "AI technique", "domain": "ml"},
    {"subject": "neural network", "predicate": "uses", "object": "gradient descent", "domain": "ml"},
    {"subject": "transformer", "predicate": "is_a", "object": "neural network architecture", "domain": "ml"},
    {"subject": "diabetes", "predicate": "is_a", "object": "metabolic disorder", "domain": "medical"},
    {"subject": "insulin", "predicate": "treats", "object": "diabetes", "domain": "medical"},
    {"subject": "COVID-19", "predicate": "caused_by", "object": "SARS-CoV-2 virus", "domain": "medical"},
    {"subject": "inflation", "predicate": "affects", "object": "purchasing power", "domain": "finance"},
    {"subject": "interest rate", "predicate": "set_by", "object": "central bank", "domain": "finance"},
    {"subject": "stock market", "predicate": "is_a", "object": "financial market", "domain": "finance"},
    {"subject": "photosynthesis", "predicate": "performed_by", "object": "plants", "domain": "general"},
    {"subject": "climate change", "predicate": "caused_by", "object": "greenhouse gas emissions", "domain": "general"},
]

# ---------------------------------------------------------------------------
# Demo corpus for keyword retrieval
# ---------------------------------------------------------------------------
_KEYWORD_CORPUS: list[dict] = [
    {"id": 1, "text": "Machine learning models require large amounts of training data.", "domain": "ml"},
    {"id": 2, "text": "Deep learning uses layered neural networks for feature extraction.", "domain": "ml"},
    {"id": 3, "text": "Insulin regulates blood glucose levels in diabetic patients.", "domain": "medical"},
    {"id": 4, "text": "Vaccines stimulate the immune system to fight specific pathogens.", "domain": "medical"},
    {"id": 5, "text": "Stock prices fluctuate based on supply and demand forces.", "domain": "finance"},
    {"id": 6, "text": "Central banks use interest rates to control inflation.", "domain": "finance"},
    {"id": 7, "text": "Renewable energy sources reduce carbon dioxide emissions.", "domain": "general"},
    {"id": 8, "text": "The water cycle involves evaporation, condensation, and precipitation.", "domain": "general"},
]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _term_vector(text: str) -> dict[str, float]:
    tokens = re.findall(r"\w+", text.lower())
    counts = Counter(tokens)
    total = sum(counts.values()) or 1
    return {t: c / total for t, c in counts.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    dot = sum(a.get(t, 0.0) * b.get(t, 0.0) for t in b)
    mag_a = math.sqrt(sum(v ** 2 for v in a.values())) or 1e-9
    mag_b = math.sqrt(sum(v ** 2 for v in b.values())) or 1e-9
    return dot / (mag_a * mag_b)


# ===========================================================================
# Tier 3 – Operational Retrieval Agents
# ===========================================================================

class VectorRetrieverAgent:
    """
    Tier-3: Dense FAISS vector similarity retrieval.

    Parameters
    ----------
    vector_store : FAISSVectorStore
        Pre-built FAISS index.
    k : int
        Number of results to return.
    """

    name = "VectorRetrieverAgent"

    def __init__(self, vector_store: FAISSVectorStore, k: int = 3) -> None:
        self.vector_store = vector_store
        self.k = k

    def retrieve(self, query: str, domain_hint: str = "") -> dict[str, Any]:
        """
        Retrieve top-k documents from FAISS.

        Parameters
        ----------
        query : str
            User query.
        domain_hint : str
            Optional domain hint (unused here, available for subclass override).

        Returns
        -------
        dict
            ``{"retriever": name, "results": list[dict], "latency": float}``
        """
        t0 = time.perf_counter()
        logger.info("[%s] FAISS search | query=%r domain_hint=%r", self.name, query, domain_hint)
        raw = self.vector_store.search(query, top_k=self.k)
        results = [
            {
                "text": item.get("text", item.get("content", str(item))),
                "score": float(item.get("score", item.get("similarity", 0.0))),
                "source": "faiss",
            }
            for item in raw
        ]
        elapsed = time.perf_counter() - t0
        return {"retriever": self.name, "results": results, "latency": round(elapsed, 4)}


class GraphRetrieverAgent:
    """
    Tier-3: Knowledge-graph triple lookup.

    Searches the simulated KG triples by cosine similarity of their
    concatenated text representation against the query.  In production
    this would use SPARQL or a dedicated graph-query language.

    Parameters
    ----------
    knowledge_graph : KnowledgeGraph
        Project knowledge-graph instance (interface only; triples from
        the local ``_KG_TRIPLES`` list are used for demonstration).
    k : int
        Number of triples to return.
    """

    name = "GraphRetrieverAgent"

    def __init__(self, knowledge_graph: KnowledgeGraph, k: int = 3) -> None:
        self.knowledge_graph = knowledge_graph
        self.k = k

    def retrieve(self, query: str, domain_hint: str = "") -> dict[str, Any]:
        """
        Look up relevant KG triples for the query.

        If ``domain_hint`` is provided the search is pre-filtered to triples
        in that domain before ranking.

        Parameters
        ----------
        query : str
            User query.
        domain_hint : str
            Optional domain label to narrow the triple search.

        Returns
        -------
        dict
            ``{"retriever": name, "results": list[dict], "latency": float}``
        """
        t0 = time.perf_counter()
        logger.info("[%s] KG lookup | query=%r domain_hint=%r", self.name, query, domain_hint)

        pool = _KG_TRIPLES
        if domain_hint:
            filtered = [t for t in pool if t.get("domain", "") == domain_hint]
            pool = filtered if filtered else pool

        q_vec = _term_vector(query)
        scored = []
        for triple in pool:
            triple_text = f"{triple['subject']} {triple['predicate']} {triple['object']}"
            t_vec = _term_vector(triple_text)
            sim = _cosine(q_vec, t_vec)
            scored.append(
                {
                    "triple": triple_text,
                    "domain": triple.get("domain", ""),
                    "score": sim,
                    "source": "knowledge_graph",
                }
            )
        scored.sort(key=lambda x: x["score"], reverse=True)

        elapsed = time.perf_counter() - t0
        return {"retriever": self.name, "results": scored[: self.k], "latency": round(elapsed, 4)}


class KeywordRetrieverAgent:
    """
    Tier-3: BM25-style keyword retrieval over the demo corpus.

    Parameters
    ----------
    k : int
        Number of results to return.
    """

    name = "KeywordRetrieverAgent"

    def __init__(self, k: int = 3) -> None:
        self.k = k

    @staticmethod
    def _bm25(query: str, text: str, avg_dl: float, k1: float = 1.5, b: float = 0.75) -> float:
        """Compute a BM25 score for a single document."""
        terms = query.lower().split()
        doc_terms = text.lower().split()
        dl = len(doc_terms)
        tf_map = Counter(doc_terms)
        score = 0.0
        for term in terms:
            tf = tf_map.get(term, 0)
            num = tf * (k1 + 1)
            den = tf + k1 * (1 - b + b * dl / avg_dl)
            score += num / (den + 1e-9)
        return score

    def retrieve(self, query: str, domain_hint: str = "") -> dict[str, Any]:
        """
        Rank the demo corpus by BM25 relevance to the query.

        Parameters
        ----------
        query : str
            User query.
        domain_hint : str
            Optional domain filter applied before ranking.

        Returns
        -------
        dict
            ``{"retriever": name, "results": list[dict], "latency": float}``
        """
        t0 = time.perf_counter()
        logger.info("[%s] BM25 search | query=%r domain_hint=%r", self.name, query, domain_hint)

        pool = _KEYWORD_CORPUS
        if domain_hint:
            filtered = [d for d in pool if d.get("domain", "") == domain_hint]
            pool = filtered if filtered else pool

        avg_dl = sum(len(d["text"].split()) for d in pool) / max(len(pool), 1)
        scored = [
            {
                "text": doc["text"],
                "domain": doc.get("domain", ""),
                "score": self._bm25(query, doc["text"], avg_dl),
                "source": "bm25_keyword",
            }
            for doc in pool
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)

        elapsed = time.perf_counter() - t0
        return {"retriever": self.name, "results": scored[: self.k], "latency": round(elapsed, 4)}


# ===========================================================================
# Tier 2 – Domain Agents
# ===========================================================================

class _BaseDomainAgent:
    """
    Shared base for Tier-2 domain agents.

    Each domain agent selects a subset of Tier-3 retrieval agents and
    delegates the query to them.

    Parameters
    ----------
    name : str
        Human-readable agent name.
    domain_key : str
        Domain label passed as ``domain_hint`` to Tier-3 agents.
    retrievers : list
        Tier-3 retriever instances to delegate to.
    """

    def __init__(self, name: str, domain_key: str, retrievers: list) -> None:
        self.name = name
        self.domain_key = domain_key
        self.retrievers = retrievers

    def process(self, query: str) -> dict[str, Any]:
        """
        Invoke all assigned Tier-3 retrievers and aggregate their results.

        Parameters
        ----------
        query : str
            User query forwarded from the Tier-1 orchestrator.

        Returns
        -------
        dict
            Aggregated results across all Tier-3 retrievers.
        """
        t0 = time.perf_counter()
        logger.info("[%s] Processing query=%r (domain=%r)", self.name, query, self.domain_key)
        retriever_outputs: dict[str, dict] = {}
        retrievers_used: list[str] = []

        for retriever in self.retrievers:
            result = retriever.retrieve(query, domain_hint=self.domain_key)
            retriever_outputs[retriever.name] = result
            retrievers_used.append(retriever.name)

        elapsed = time.perf_counter() - t0
        return {
            "domain_agent": self.name,
            "domain": self.domain_key,
            "retrievers_used": retrievers_used,
            "retriever_outputs": retriever_outputs,
            "latency": round(elapsed, 4),
        }


class MedicalDomainAgent(_BaseDomainAgent):
    """
    Tier-2 domain agent specialised in medical and health queries.

    Uses ``VectorRetrieverAgent`` (for dense semantic retrieval) and
    ``GraphRetrieverAgent`` (for medical ontology triples).
    """

    def __init__(
        self,
        vector_retriever: VectorRetrieverAgent,
        graph_retriever: GraphRetrieverAgent,
    ) -> None:
        super().__init__(
            name="MedicalDomainAgent",
            domain_key="medical",
            retrievers=[vector_retriever, graph_retriever],
        )


class FinanceDomainAgent(_BaseDomainAgent):
    """
    Tier-2 domain agent specialised in finance and economics queries.

    Uses ``KeywordRetrieverAgent`` (structured financial terminology) and
    ``GraphRetrieverAgent`` (financial KG triples).
    """

    def __init__(
        self,
        keyword_retriever: KeywordRetrieverAgent,
        graph_retriever: GraphRetrieverAgent,
    ) -> None:
        super().__init__(
            name="FinanceDomainAgent",
            domain_key="finance",
            retrievers=[keyword_retriever, graph_retriever],
        )


class GeneralDomainAgent(_BaseDomainAgent):
    """
    Tier-2 catch-all domain agent for queries that do not belong to a
    specific specialist domain.

    Uses all three Tier-3 retrievers: ``VectorRetrieverAgent``,
    ``GraphRetrieverAgent``, and ``KeywordRetrieverAgent``.
    """

    def __init__(
        self,
        vector_retriever: VectorRetrieverAgent,
        graph_retriever: GraphRetrieverAgent,
        keyword_retriever: KeywordRetrieverAgent,
    ) -> None:
        super().__init__(
            name="GeneralDomainAgent",
            domain_key="general",
            retrievers=[vector_retriever, graph_retriever, keyword_retriever],
        )


# ===========================================================================
# Tier 1 – Strategic Orchestrator
# ===========================================================================

_ORCHESTRATOR_PROMPT = """You are a strategic orchestrator for a Hierarchical Retrieval-Augmented Generation (RAG) system.
Analyse the user's question and select which domain agents should handle it.

Available domain agents:
  - medical   : Questions about health, medicine, diseases, treatments, anatomy.
  - finance   : Questions about money, stocks, banking, economics, investment.
  - general   : All other questions, or when multiple domains apply.

Rules:
  - You may activate MORE THAN ONE domain (comma-separated).
  - Always activate "general" if the query spans multiple domains or is ambiguous.

User question: "{query}"

Respond with a JSON object ONLY:
{{
  "domains": ["<domain1>", "<domain2>"],
  "reasoning": "<one sentence rationale>"
}}
"""


class StrategicOrchestratorAgent:
    """
    Tier-1: Strategic Orchestrator.

    Analyses the incoming query and decides which Tier-2 domain agents
    are relevant.  The decision is made by the LLM via a structured prompt.

    Parameters
    ----------
    llm : LocalLLM
        Language model used for domain classification and answer synthesis.
    domain_agents : dict[str, _BaseDomainAgent]
        Mapping from domain name (e.g. ``"medical"``) to the Tier-2 agent.
    """

    VALID_DOMAINS = {"medical", "finance", "general"}

    def __init__(self, llm: LocalLLM, domain_agents: dict[str, _BaseDomainAgent]) -> None:
        self.llm = llm
        self.domain_agents = domain_agents

    def classify_domains(self, query: str) -> tuple[list[str], str]:
        """
        Ask the LLM to classify the query into one or more domains.

        Parameters
        ----------
        query : str
            User query.

        Returns
        -------
        tuple[list[str], str]
            ``(selected_domains, reasoning)``
        """
        import json

        prompt = _ORCHESTRATOR_PROMPT.format(query=query)
        raw = self.llm.generate(prompt)

        try:
            json_match = re.search(r"\{.*?\}", raw, re.DOTALL)
            decision = json.loads(json_match.group() if json_match else raw)
            domains = [d.strip().lower() for d in decision.get("domains", ["general"])]
            domains = [d for d in domains if d in self.VALID_DOMAINS] or ["general"]
            reasoning = decision.get("reasoning", "")
        except Exception:  # noqa: BLE001
            domains = ["general"]
            reasoning = "Fallback: could not parse LLM response."

        logger.info(
            "[StrategicOrchestrator] domains=%s | reasoning=%r", domains, reasoning
        )
        return domains, reasoning

    def synthesize(
        self,
        query: str,
        domain_outputs: dict[str, dict],
    ) -> str:
        """
        Compose a final answer from the outputs of all activated domain agents.

        Parameters
        ----------
        query : str
            Original user question.
        domain_outputs : dict[str, dict]
            Results from each activated Tier-2 domain agent.

        Returns
        -------
        str
            LLM-generated synthesized answer.
        """
        sections: list[str] = []
        for domain_name, output in domain_outputs.items():
            for retriever_name, r_output in output.get("retriever_outputs", {}).items():
                docs = r_output.get("results", [])
                snippets = "\n".join(
                    f"  • {d.get('text', d.get('triple', ''))}" for d in docs
                )
                if snippets:
                    sections.append(f"[{domain_name} / {retriever_name}]\n{snippets}")

        context_str = "\n\n".join(sections) or "No context retrieved."
        prompt = (
            "You are a knowledgeable assistant. Use the context from the hierarchical"
            " RAG system to answer the question comprehensively.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        return self.llm.generate(prompt)


# ===========================================================================
# Public facade: HierarchicalRAG
# ===========================================================================

class HierarchicalRAG:
    """
    Hierarchical Agentic RAG (§5.3).

    Implements a 3-tier agent hierarchy:

    * **Tier 1** – ``StrategicOrchestratorAgent``: classifies query domains.
    * **Tier 2** – Domain agents (Medical / Finance / General): decide which
      retrievers to call.
    * **Tier 3** – Retriever agents (Vector / Graph / Keyword): fetch documents.

    Parameters
    ----------
    llm : LocalLLM
        Language model used at Tiers 1 and 2 for decision-making and synthesis.
    vector_store : FAISSVectorStore
        FAISS index for the ``VectorRetrieverAgent``.
    knowledge_graph : KnowledgeGraph
        Graph store for the ``GraphRetrieverAgent``.
    k : int
        Number of documents each Tier-3 retriever should return (default 3).
    """

    def __init__(
        self,
        llm: LocalLLM,
        vector_store: FAISSVectorStore,
        knowledge_graph: KnowledgeGraph,
        k: int = 3,
    ) -> None:
        self.llm = llm
        self.k = k

        # ---- Tier 3: Retrieval Primitives ----
        vector_ret = VectorRetrieverAgent(vector_store=vector_store, k=k)
        graph_ret = GraphRetrieverAgent(knowledge_graph=knowledge_graph, k=k)
        keyword_ret = KeywordRetrieverAgent(k=k)

        # ---- Tier 2: Domain Agents ----
        medical_agent = MedicalDomainAgent(
            vector_retriever=vector_ret, graph_retriever=graph_ret
        )
        finance_agent = FinanceDomainAgent(
            keyword_retriever=keyword_ret, graph_retriever=graph_ret
        )
        general_agent = GeneralDomainAgent(
            vector_retriever=vector_ret,
            graph_retriever=graph_ret,
            keyword_retriever=keyword_ret,
        )

        self._domain_agents: dict[str, _BaseDomainAgent] = {
            "medical": medical_agent,
            "finance": finance_agent,
            "general": general_agent,
        }

        # ---- Tier 1: Strategic Orchestrator ----
        self._orchestrator = StrategicOrchestratorAgent(
            llm=llm, domain_agents=self._domain_agents
        )
        logger.info("HierarchicalRAG initialised (k=%d, domains=%s)", k, list(self._domain_agents))

    def query(self, question: str) -> dict:
        """
        Execute the full 3-tier hierarchical retrieval pipeline.

        The cascade proceeds as follows:

        1. **Tier 1**: Orchestrator classifies the query into domain(s).
        2. **Tier 2**: Selected domain agents invoke their Tier-3 retrievers.
        3. **Tier 3**: Retrievers fetch relevant documents/triples.
        4. **Synthesis**: Orchestrator composes the final answer from all results.

        Parameters
        ----------
        question : str
            The user's natural-language question.

        Returns
        -------
        dict
            Result dictionary containing:

            * ``tier1_decision``       – domain classification and reasoning.
            * ``tier2_domains_activated`` – list of activated domain agent names.
            * ``tier3_retrievers_used`` – flat list of retriever names used.
            * ``domain_outputs``       – per-domain raw retrieval outputs.
            * ``answer``               – synthesized answer.
            * ``latency``              – total wall-clock latency in seconds.
        """
        logger.info("=== HierarchicalRAG.query | question=%r ===", question)
        t_start = time.perf_counter()

        # ------ Tier 1: Domain classification ------
        selected_domains, tier1_reasoning = self._orchestrator.classify_domains(question)
        tier1_decision = {
            "selected_domains": selected_domains,
            "reasoning": tier1_reasoning,
        }
        logger.info("Tier 1 decision: %s", tier1_decision)

        # ------ Tier 2 + Tier 3: Domain agents activate retrievers ------
        domain_outputs: dict[str, dict] = {}
        all_retrievers_used: list[str] = []

        for domain in selected_domains:
            agent = self._domain_agents.get(domain)
            if agent is None:
                logger.warning("No agent registered for domain %r; skipping.", domain)
                continue
            output = agent.process(question)
            domain_outputs[domain] = output
            all_retrievers_used.extend(output.get("retrievers_used", []))
            logger.info(
                "Tier 2 (%s) activated Tier-3 retrievers: %s",
                domain,
                output.get("retrievers_used", []),
            )

        # ------ Synthesis ------
        answer = self._orchestrator.synthesize(question, domain_outputs)

        latency = time.perf_counter() - t_start
        logger.info("HierarchicalRAG.query completed in %.3f s", latency)

        return {
            "tier1_decision": tier1_decision,
            "tier2_domains_activated": selected_domains,
            "tier3_retrievers_used": list(dict.fromkeys(all_retrievers_used)),  # deduped, ordered
            "domain_outputs": domain_outputs,
            "answer": answer,
            "latency": round(latency, 4),
        }
