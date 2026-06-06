"""
Parallelization — §4.3 of 'Agentic RAG: A Survey' (arXiv:2501.09136)
=====================================================================
Implements concurrent parallel worker execution with two aggregation
strategies:

- **Voting aggregation** — majority vote over independent answers
  (best for factoid questions where each worker produces a short answer).

- **Sectioning aggregation** — each worker is responsible for a distinct
  topic section; results are merged into a structured report
  (best for broad queries requiring multi-angle coverage).

The parallelism is achieved via Python's ``concurrent.futures.ThreadPoolExecutor``
for I/O-bound LLM calls.

References
----------
- Survey §4.3: Parallelisation as a RAG workflow pattern.
"""

from __future__ import annotations

import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from core.llm import LocalLLM
    from core.vector_store import FAISSVectorStore


# ---------------------------------------------------------------------------
# Default pre-built workers
# ---------------------------------------------------------------------------

def _make_dense_retrieval_worker(llm: "LocalLLM", vector_store: "FAISSVectorStore") -> Callable[[str], dict]:
    """
    Worker: dense vector retrieval + generation.

    Performs FAISS similarity search and generates an answer from the
    retrieved context.

    Parameters
    ----------
    llm : LocalLLM
        Language model for generation.
    vector_store : FAISSVectorStore
        Vector store for retrieval.

    Returns
    -------
    Callable[[str], dict]
        A callable that takes a query and returns ``{"section": str, "answer": str}``.
    """

    def _fn(query: str) -> dict:
        docs = vector_store.search(query, top_k=5)
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
            "Using only the context, provide a concise, accurate answer:"
        )
        answer = llm.generate(prompt).strip()
        return {"section": "Vector Retrieval", "answer": answer}

    return _fn


def _make_cot_worker(llm: "LocalLLM") -> Callable[[str], dict]:
    """
    Worker: chain-of-thought reasoning without retrieval.

    Prompts the LLM to reason step-by-step using its parametric knowledge.

    Parameters
    ----------
    llm : LocalLLM
        Language model for reasoning.

    Returns
    -------
    Callable[[str], dict]
        A callable that takes a query and returns ``{"section": str, "answer": str}``.
    """

    def _fn(query: str) -> dict:
        prompt = (
            f"Question: {query}\n\n"
            "Think through this carefully, step by step, using your knowledge:\n"
            "Step 1: Identify the key aspects of the question.\n"
            "Step 2: Recall relevant information.\n"
            "Step 3: Reason to a conclusion.\n\n"
            "Final Answer:"
        )
        answer = llm.generate(prompt).strip()
        return {"section": "Chain-of-Thought Reasoning", "answer": answer}

    return _fn


def _make_perspective_worker(llm: "LocalLLM", perspective: str) -> Callable[[str], dict]:
    """
    Worker: answer the query from a specific analytical perspective.

    Parameters
    ----------
    llm : LocalLLM
        Language model.
    perspective : str
        The lens from which to analyse the query (e.g. 'technical', 'ethical').

    Returns
    -------
    Callable[[str], dict]
        A callable that takes a query and returns ``{"section": str, "answer": str}``.
    """

    def _fn(query: str) -> dict:
        prompt = (
            f"Analyse the following question from a {perspective} perspective:\n\n"
            f"Question: {query}\n\n"
            f"Provide a focused, insightful analysis from the {perspective} viewpoint:"
        )
        answer = llm.generate(prompt).strip()
        return {"section": f"{perspective.capitalize()} Perspective", "answer": answer}

    return _fn


# ---------------------------------------------------------------------------
# ParallelWorkflow
# ---------------------------------------------------------------------------

class ParallelWorkflow:
    """
    Implements the Parallelisation workflow pattern for RAG (§4.3).

    Multiple worker functions run concurrently on the same query; their
    outputs are then aggregated via voting or sectioning strategies.

    Parameters
    ----------
    llm : LocalLLM
        Language model used by workers and for final synthesis.
    vector_store : FAISSVectorStore
        Vector store for retrieval-based workers.
    max_workers : int, optional
        Maximum thread-pool concurrency (default 4).
    timeout : float, optional
        Per-worker execution timeout in seconds (default 30.0).
    """

    def __init__(
        self,
        llm: "LocalLLM",
        vector_store: "FAISSVectorStore",
        max_workers: int = 4,
        timeout: float = 30.0,
    ) -> None:
        self.llm = llm
        self.vector_store = vector_store
        self.max_workers = max_workers
        self.timeout = timeout
        self._workers: dict[str, Callable[[str], dict | str]] = {}

        # Register three default workers
        self.add_worker("dense_retrieval", _make_dense_retrieval_worker(llm, vector_store))
        self.add_worker("chain_of_thought", _make_cot_worker(llm))
        self.add_worker("technical_perspective", _make_perspective_worker(llm, "technical"))

    # ------------------------------------------------------------------
    # Worker management
    # ------------------------------------------------------------------

    def add_worker(self, name: str, fn: Callable[[str], dict | str]) -> None:
        """
        Register a worker function.

        Worker functions receive a query string and return either:
        - A ``str`` (used directly as the answer), or
        - A ``dict`` with at least an ``"answer"`` key and optionally a
          ``"section"`` key (used for sectioning aggregation).

        Parameters
        ----------
        name : str
            Unique name for the worker.
        fn : Callable[[str], dict | str]
            The worker function.
        """
        self._workers[name] = fn

    def remove_worker(self, name: str) -> None:
        """Remove a worker by name."""
        self._workers.pop(name, None)

    # ------------------------------------------------------------------
    # Aggregation strategies
    # ------------------------------------------------------------------

    def _aggregate_voting(self, results: list[str]) -> str:
        """
        Select the most common answer from a list of candidate answers (majority vote).

        Uses exact string matching as the primary voting mechanism; if all
        answers are unique, falls back to returning the longest / most
        detailed answer as the presumed most complete response.

        Parameters
        ----------
        results : list[str]
            A list of answer strings, one per worker.

        Returns
        -------
        str
            The winning answer.
        """
        if not results:
            return "No results to aggregate."

        # Normalise (lower-strip) for counting
        normalised = [r.lower().strip() for r in results]
        counts = Counter(normalised)
        most_common_normalised, top_count = counts.most_common(1)[0]

        if top_count > 1:
            # Return the original-case version of the most common answer
            for original in results:
                if original.lower().strip() == most_common_normalised:
                    return original

        # All unique — return the most detailed answer (longest)
        return max(results, key=len)

    def _aggregate_sectioning(self, results: list[dict]) -> str:
        """
        Merge section-keyed answers from different workers into a structured report.

        Each result dict is expected to have a ``"section"`` key (section title)
        and an ``"answer"`` key (the content for that section).

        A final LLM synthesis step produces a cohesive combined answer.

        Parameters
        ----------
        results : list[dict]
            Each dict must contain ``"section"`` and ``"answer"`` keys.

        Returns
        -------
        str
            A structured, merged answer combining all sections.
        """
        if not results:
            return "No results to aggregate."

        # Build section-based markdown
        sections = []
        for r in results:
            section_title = r.get("section", "Analysis")
            answer = r.get("answer", "")
            sections.append(f"### {section_title}\n{answer}")

        sections_text = "\n\n".join(sections)

        # LLM synthesis for a cohesive final answer
        synthesis_prompt = (
            "You are a report synthesiser. Below are answers from multiple specialised "
            "workers, each covering a different aspect of the query.\n\n"
            f"{sections_text}\n\n"
            "Synthesise all of the above into a single, coherent, well-structured "
            "comprehensive answer. Preserve the key insights from each section."
            "\n\nSynthesised Answer:"
        )
        synthesised = self.llm.generate(synthesis_prompt).strip()

        # Return both the structured sections and the synthesis
        return f"{sections_text}\n\n---\n\n### Synthesised Answer\n{synthesised}"

    # ------------------------------------------------------------------
    # Parallel execution
    # ------------------------------------------------------------------

    def run_parallel(
        self,
        query: str,
        aggregation: str = "sectioning",
    ) -> dict:
        """
        Execute all registered workers concurrently and aggregate results.

        Parameters
        ----------
        query : str
            The user question passed to every worker.
        aggregation : str, optional
            Aggregation strategy: ``'voting'`` or ``'sectioning'``
            (default ``'sectioning'``).

        Returns
        -------
        dict
            {
              "worker_results"    : dict[str, dict|str],  # per-worker outputs
              "aggregated_answer" : str,                  # aggregated final answer
              "failed_workers"    : list[str],            # names of failed workers
              "latency"           : float,                # total wall-clock time
              "aggregation_mode"  : str,                  # strategy used
            }

        Raises
        ------
        ValueError
            If ``aggregation`` is not ``'voting'`` or ``'sectioning'``.
        """
        if aggregation not in ("voting", "sectioning"):
            raise ValueError(
                f"aggregation must be 'voting' or 'sectioning', got '{aggregation}'"
            )

        t_start = time.perf_counter()
        worker_results: dict[str, dict | str] = {}
        failed_workers: list[str] = []

        # Submit all workers to thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_name = {
                executor.submit(fn, query): name
                for name, fn in self._workers.items()
            }

            for future in as_completed(future_to_name, timeout=self.timeout):
                name = future_to_name[future]
                try:
                    result = future.result()
                    worker_results[name] = result
                except Exception as exc:
                    failed_workers.append(name)
                    worker_results[name] = {
                        "section": name,
                        "answer": f"[Worker failed: {exc}]",
                        "error": str(exc),
                    }

        # Aggregate results
        if aggregation == "voting":
            # Extract answer strings for voting
            answers = []
            for name, result in worker_results.items():
                if isinstance(result, str):
                    answers.append(result)
                elif isinstance(result, dict):
                    answers.append(result.get("answer", ""))
            aggregated_answer = self._aggregate_voting(answers)
        else:
            # Extract section dicts for sectioning
            section_results = []
            for name, result in worker_results.items():
                if isinstance(result, str):
                    section_results.append({"section": name, "answer": result})
                elif isinstance(result, dict):
                    r = result.copy()
                    r.setdefault("section", name)
                    section_results.append(r)
            aggregated_answer = self._aggregate_sectioning(section_results)

        latency = time.perf_counter() - t_start

        return {
            "worker_results": worker_results,
            "aggregated_answer": aggregated_answer,
            "failed_workers": failed_workers,
            "latency": round(latency, 4),
            "aggregation_mode": aggregation,
        }
