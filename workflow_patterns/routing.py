"""
Routing — §4.2 of 'Agentic RAG: A Survey' (arXiv:2501.09136)
=============================================================
Implements an intelligent query-routing workflow that classifies an
incoming query into one of several categories and dispatches it to the
most appropriate handler.

Query types supported:
- ``'factoid'``        — short, single-fact questions → dense vector search
- ``'complex'``        — multi-hop / analytical questions → planning agent
- ``'sql'``            — structured data / tabular questions → SQL-style handler
- ``'conversational'`` — chat / follow-up queries → conversational handler

The ``QueryRouter`` classifies using the LLM; the ``RoutingWorkflow``
manages handler registration and dispatches accordingly.

References
----------
- Survey §4.2: Routing as a RAG workflow pattern for query-adaptive pipelines.
"""

from __future__ import annotations

import json
import re
import time
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from core.llm import LocalLLM
    from core.vector_store import FAISSVectorStore


# ---------------------------------------------------------------------------
# Supported query types
# ---------------------------------------------------------------------------

SUPPORTED_QUERY_TYPES = ("factoid", "complex", "sql", "conversational")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_CLASSIFY_PROMPT = """You are an expert at classifying user queries for a RAG (Retrieval-Augmented Generation) system.

Classify the following query into EXACTLY ONE of these categories:
- factoid: Simple, direct questions with short answers (e.g. "What year was X founded?", "Who invented Y?")
- complex: Multi-step, analytical, or comparative questions requiring reasoning (e.g. "Compare X and Y in terms of Z", "Why did X lead to Y?")
- sql: Questions about structured/tabular data, statistics, or database records (e.g. "How many users signed up in Q1?", "What is the average revenue per region?")
- conversational: Casual, follow-up, or clarification queries (e.g. "Can you explain that again?", "What do you mean by X?")

Query: {query}

Respond ONLY with valid JSON:
{{
  "query_type": "<one of: factoid, complex, sql, conversational>",
  "confidence": <0.0–1.0>,
  "reasoning": "<brief explanation>"
}}"""

# ---------------------------------------------------------------------------
# Default handler implementations
# ---------------------------------------------------------------------------

def _default_factoid_handler(
    query: str,
    llm: "LocalLLM",
    vector_store: "FAISSVectorStore",
    top_k: int = 3,
) -> str:
    """
    Handle factoid queries with simple vector retrieval + direct generation.

    Retrieves the top-k most relevant document chunks and prompts the LLM
    to extract a concise factual answer.

    Parameters
    ----------
    query : str
        The factoid question.
    llm : LocalLLM
        Language model for generation.
    vector_store : FAISSVectorStore
        Vector store for retrieval.
    top_k : int, optional
        Number of results to retrieve.

    Returns
    -------
    str
        A concise factual answer.
    """
    docs = vector_store.search(query, top_k=top_k)
    chunks = []
    for doc in docs:
        if isinstance(doc, str):
            chunks.append(doc)
        elif hasattr(doc, "text"):
            chunks.append(doc.text)
        elif hasattr(doc, "page_content"):
            chunks.append(doc.page_content)
        else:
            chunks.append(str(doc))
    context = "\n\n".join(chunks)

    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer concisely and directly in 1–2 sentences:\n"
    )
    return llm.generate(prompt).strip()


def _default_complex_handler(
    query: str,
    llm: "LocalLLM",
    vector_store: "FAISSVectorStore",
    top_k: int = 5,
) -> str:
    """
    Handle complex multi-hop queries with chain-of-thought prompting.

    Retrieves more documents and instructs the LLM to reason step-by-step
    before producing a final answer.

    Parameters
    ----------
    query : str
        The complex question.
    llm : LocalLLM
        Language model for generation.
    vector_store : FAISSVectorStore
        Vector store for retrieval.
    top_k : int, optional
        Number of results to retrieve.

    Returns
    -------
    str
        A detailed reasoned answer.
    """
    docs = vector_store.search(query, top_k=top_k)
    chunks = []
    for doc in docs:
        if isinstance(doc, str):
            chunks.append(doc)
        elif hasattr(doc, "text"):
            chunks.append(doc.text)
        elif hasattr(doc, "page_content"):
            chunks.append(doc.page_content)
        else:
            chunks.append(str(doc))
    context = "\n\n".join(chunks)

    prompt = (
        f"Context:\n{context}\n\n"
        f"Complex Question: {query}\n\n"
        "Think through this step-by-step:\n"
        "1. Identify the key components of the question.\n"
        "2. Address each component using the context.\n"
        "3. Synthesise a comprehensive final answer.\n\n"
        "Final Answer:"
    )
    return llm.generate(prompt).strip()


def _default_sql_handler(query: str, llm: "LocalLLM", **_kwargs) -> str:
    """
    Handle SQL/structured-data queries by generating a SQL query and explanation.

    In production this would connect to a database; here it demonstrates the
    SQL generation capability using the LLM.

    Parameters
    ----------
    query : str
        The structured/tabular question.
    llm : LocalLLM
        Language model for SQL generation.

    Returns
    -------
    str
        A generated SQL query with a natural-language explanation.
    """
    prompt = (
        f"The user has a question about structured/tabular data: '{query}'\n\n"
        "1. Generate an appropriate SQL query that would answer this question "
        "(assume a generic schema with common business tables: users, orders, products, transactions).\n"
        "2. Explain what the query does in plain English.\n\n"
        "SQL Query:\n```sql\n"
    )
    raw = llm.generate(prompt).strip()
    return f"[SQL Handler]\n{raw}"


def _default_conversational_handler(query: str, llm: "LocalLLM", **_kwargs) -> str:
    """
    Handle conversational / follow-up queries with an empathetic, natural response.

    Parameters
    ----------
    query : str
        The conversational query.
    llm : LocalLLM
        Language model for generation.

    Returns
    -------
    str
        A natural, conversational response.
    """
    prompt = (
        f"The user says: '{query}'\n\n"
        "This is a conversational message. Respond naturally, helpfully, and concisely. "
        "If it's a clarification request, ask a focused follow-up question.\n\n"
        "Response:"
    )
    return llm.generate(prompt).strip()


# ---------------------------------------------------------------------------
# QueryRouter
# ---------------------------------------------------------------------------

class QueryRouter:
    """
    Classifies a query into one of the supported query types using the LLM.

    Parameters
    ----------
    llm : LocalLLM
        Language model used for classification.
    """

    def __init__(self, llm: "LocalLLM") -> None:
        self.llm = llm

    def _classify_query(self, query: str) -> str:
        """
        Use the LLM to classify the query into a supported category.

        Prompts the LLM for a JSON response, parses it, and validates
        the returned category.  Falls back to ``'factoid'`` if parsing fails.

        Parameters
        ----------
        query : str
            The user question to classify.

        Returns
        -------
        str
            One of: ``'factoid'``, ``'complex'``, ``'sql'``, ``'conversational'``.
        """
        prompt = _CLASSIFY_PROMPT.format(query=query)
        raw = self.llm.generate(prompt).strip()

        # Parse JSON response
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                query_type = data.get("query_type", "").lower().strip()
                if query_type in SUPPORTED_QUERY_TYPES:
                    return query_type
            except json.JSONDecodeError:
                pass

        # Keyword-based fallback classification
        query_lower = query.lower()
        if any(w in query_lower for w in ("select", "count", "average", "sum", "table", "database", "records")):
            return "sql"
        if any(w in query_lower for w in ("compare", "contrast", "why did", "analyse", "analyze", "explain how", "relationship between")):
            return "complex"
        if any(w in query_lower for w in ("what do you mean", "can you explain", "tell me more", "again", "clarify")):
            return "conversational"
        return "factoid"

    def classify(self, query: str) -> dict:
        """
        Classify a query and return both the type and metadata.

        Parameters
        ----------
        query : str
            The user question.

        Returns
        -------
        dict
            {
              "query_type" : str,   # classified category
              "query"      : str,   # original query
            }
        """
        query_type = self._classify_query(query)
        return {"query_type": query_type, "query": query}


# ---------------------------------------------------------------------------
# RoutingWorkflow
# ---------------------------------------------------------------------------

class RoutingWorkflow:
    """
    Manages query routing: classifies queries and dispatches to registered handlers.

    Parameters
    ----------
    llm : LocalLLM
        Language model for classification and default handlers.
    vector_store : FAISSVectorStore
        Vector store passed to handlers that perform retrieval.
    """

    def __init__(self, llm: "LocalLLM", vector_store: "FAISSVectorStore") -> None:
        self.llm = llm
        self.vector_store = vector_store
        self.router = QueryRouter(llm)
        self._routes: dict[str, Callable] = {}

        # Register default handlers for all supported query types
        self.register_route(
            "factoid",
            lambda q: _default_factoid_handler(q, self.llm, self.vector_store),
        )
        self.register_route(
            "complex",
            lambda q: _default_complex_handler(q, self.llm, self.vector_store),
        )
        self.register_route(
            "sql",
            lambda q: _default_sql_handler(q, self.llm),
        )
        self.register_route(
            "conversational",
            lambda q: _default_conversational_handler(q, self.llm),
        )

    def register_route(self, query_type: str, handler: Callable[[str], str]) -> None:
        """
        Register or replace a handler for a given query type.

        Parameters
        ----------
        query_type : str
            The query category to handle (must be one of the supported types).
        handler : Callable[[str], str]
            A callable that accepts a query string and returns a string answer.

        Raises
        ------
        ValueError
            If ``query_type`` is not in ``SUPPORTED_QUERY_TYPES``.
        """
        if query_type not in SUPPORTED_QUERY_TYPES:
            raise ValueError(
                f"Unsupported query type '{query_type}'. "
                f"Must be one of: {SUPPORTED_QUERY_TYPES}"
            )
        self._routes[query_type] = handler

    def query(self, question: str) -> dict:
        """
        Classify the question and dispatch it to the appropriate handler.

        Parameters
        ----------
        question : str
            The user question.

        Returns
        -------
        dict
            {
              "query_type"   : str,   # classified type
              "handler_used" : str,   # name/label of the invoked handler
              "answer"       : str,   # final answer from the handler
              "latency"      : float, # wall-clock time in seconds
            }
        """
        t_start = time.perf_counter()

        # Step 1: Classify
        classification = self.router.classify(question)
        query_type = classification["query_type"]

        # Step 2: Dispatch
        handler = self._routes.get(query_type)
        if handler is None:
            # Fallback to factoid
            query_type = "factoid"
            handler = self._routes["factoid"]

        answer = handler(question)
        latency = time.perf_counter() - t_start

        return {
            "query_type": query_type,
            "handler_used": f"{query_type}_handler",
            "answer": answer,
            "latency": round(latency, 4),
        }
