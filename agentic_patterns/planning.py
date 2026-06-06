"""
Planning Pattern — §3.2 of 'Agentic RAG: A Survey' (arXiv:2501.09136)
=======================================================================
Implements a task-decomposition planning loop where a complex query is
broken into 2–4 independently answerable sub-queries, each sub-query is
solved with dedicated retrieval, and the sub-answers are synthesised into
a coherent final response.

This mirrors approaches such as:
- Least-to-Most Prompting (Zhou et al., 2023)
- HyDE / step-back prompting for multi-hop QA
- Survey §3.2: Planning as an agentic design pattern in RAG pipelines.
"""

from __future__ import annotations

import json
import re
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.llm import LocalLLM
    from core.vector_store import FAISSVectorStore


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_DECOMPOSE_PROMPT = """You are an expert at breaking complex questions into simpler sub-questions.

Given the following complex question, decompose it into 2 to 4 specific, self-contained sub-questions that together cover all aspects needed to answer the original question.

Complex Question: {query}

Rules:
- Each sub-question must be answerable independently.
- Sub-questions should together provide all information needed to answer the original.
- Keep sub-questions concise and specific.
- Number them 1, 2, 3 ...

Output ONLY a JSON array of strings, e.g.:
["sub-question 1", "sub-question 2", "sub-question 3"]"""

_SUBQUERY_PROMPT = """You are a knowledgeable assistant. Use the provided context to answer the specific question accurately and concisely.

Context:
{context}

Question: {subquery}

Answer:"""

_MERGE_PROMPT = """You are an expert at synthesising information from multiple sources into a comprehensive answer.

Original Question: {query}

The following sub-questions were answered to address the original question:

{sub_results_text}

Using all of the above information, write a comprehensive, well-structured answer to the original question. Do not reference "sub-question" in your answer — write as if you are directly answering the user.

Final Answer:"""


# ---------------------------------------------------------------------------
# PlanningAgent
# ---------------------------------------------------------------------------

class PlanningAgent:
    """
    Implements the Planning agentic design pattern for RAG (§3.2).

    The agent:
    1. Decomposes a complex query into 2–4 sub-queries via LLM.
    2. Solves each sub-query independently using vector-store retrieval.
    3. Merges all sub-answers into a single coherent final response.

    Parameters
    ----------
    llm : LocalLLM
        Language model used for decomposition, solving, and merging.
    vector_store : FAISSVectorStore
        Vector store used for retrieval at each sub-query step.
    top_k : int, optional
        Number of documents to retrieve per sub-query (default 4).
    max_subqueries : int, optional
        Upper bound on number of sub-queries generated (default 4).
    """

    def __init__(
        self,
        llm: "LocalLLM",
        vector_store: "FAISSVectorStore",
        top_k: int = 4,
        max_subqueries: int = 4,
    ) -> None:
        self.llm = llm
        self.vector_store = vector_store
        self.top_k = top_k
        self.max_subqueries = max_subqueries

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _retrieve_context(self, query: str) -> str:
        """
        Retrieve top-k document chunks for a given query.

        Parameters
        ----------
        query : str
            Query string for vector-store search.

        Returns
        -------
        str
            Concatenated retrieved document chunks.
        """
        docs = self.vector_store.search(query, top_k=self.top_k)
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
        return "\n\n".join(chunks)

    def _decompose_task(self, query: str) -> list[str]:
        """
        Use the LLM to break a complex query into 2–4 sub-queries.

        The model is prompted to return a JSON array of strings.  If the
        response cannot be parsed, a simple fallback splits the query into
        two generic sub-questions.

        Parameters
        ----------
        query : str
            The original complex user question.

        Returns
        -------
        list[str]
            A list of 2–4 sub-query strings.
        """
        prompt = _DECOMPOSE_PROMPT.format(query=query)
        raw = self.llm.generate(prompt).strip()

        # Try to extract a JSON array from the response
        json_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if json_match:
            try:
                sub_queries = json.loads(json_match.group())
                if isinstance(sub_queries, list) and sub_queries:
                    # Enforce max limit and string type
                    sub_queries = [str(sq).strip() for sq in sub_queries if sq]
                    return sub_queries[: self.max_subqueries]
            except json.JSONDecodeError:
                pass

        # Fallback: numbered lines parsing
        lines = [
            re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            for line in raw.splitlines()
            if re.match(r"^\d+[\.\)]", line.strip())
        ]
        if lines:
            return lines[: self.max_subqueries]

        # Last-resort: treat the original query as a single sub-query
        return [query]

    def _solve_subquery(self, subquery: str, context: str) -> str:
        """
        Answer a single sub-query given retrieved context.

        Parameters
        ----------
        subquery : str
            The specific sub-question to answer.
        context : str
            Concatenated retrieved documents for this sub-query.

        Returns
        -------
        str
            The generated answer for this sub-query.
        """
        prompt = _SUBQUERY_PROMPT.format(context=context, subquery=subquery)
        return self.llm.generate(prompt).strip()

    def _merge_results(self, query: str, sub_results: list[dict]) -> str:
        """
        Synthesise all sub-query answers into a coherent final response.

        Parameters
        ----------
        query : str
            The original user question.
        sub_results : list[dict]
            Each dict contains ``subquery`` (str) and ``answer`` (str).

        Returns
        -------
        str
            The synthesised final answer.
        """
        sub_results_text = "\n\n".join(
            f"Sub-question {i + 1}: {r['subquery']}\nAnswer: {r['answer']}"
            for i, r in enumerate(sub_results)
        )
        prompt = _MERGE_PROMPT.format(
            query=query, sub_results_text=sub_results_text
        )
        return self.llm.generate(prompt).strip()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def query(self, question: str) -> dict:
        """
        Execute the full planning pipeline for a complex question.

        Workflow
        --------
        1. Decompose the question into sub-queries.
        2. For each sub-query: retrieve context → generate sub-answer.
        3. Merge all sub-answers into a final comprehensive response.

        Parameters
        ----------
        question : str
            The complex user question to answer.

        Returns
        -------
        dict
            {
              "plan"         : list[str],  # the sub-queries (the plan)
              "sub_results"  : list[dict], # per-sub-query results
              "final_answer" : str,        # synthesised final answer
              "latency"      : float,      # wall-clock time in seconds
            }

        Each ``sub_results`` entry::

            {
              "subquery" : str,
              "context"  : str,   # retrieved context (truncated for display)
              "answer"   : str,
            }
        """
        t_start = time.perf_counter()

        # Step 1: Decompose
        plan = self._decompose_task(question)

        # Step 2: Solve each sub-query
        sub_results: list[dict] = []
        for subquery in plan:
            context = self._retrieve_context(subquery)
            answer = self._solve_subquery(subquery, context)
            sub_results.append(
                {
                    "subquery": subquery,
                    # Truncate context to keep the output dict readable
                    "context_preview": context[:500] + ("…" if len(context) > 500 else ""),
                    "answer": answer,
                }
            )

        # Step 3: Merge
        final_answer = self._merge_results(question, sub_results)

        latency = time.perf_counter() - t_start

        return {
            "plan": plan,
            "sub_results": sub_results,
            "final_answer": final_answer,
            "latency": round(latency, 4),
        }
