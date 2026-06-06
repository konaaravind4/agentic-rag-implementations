"""
Orchestrator-Workers — §4.4 of 'Agentic RAG: A Survey' (arXiv:2501.09136)
===========================================================================
Implements an LLM-driven orchestrator that dynamically decomposes a complex
query into a task list, dispatches sub-tasks to specialised ``WorkerAgent``
instances, and compiles the results into a coherent final answer.

Unlike ``PlanningAgent`` (which decomposes into sub-queries), the
orchestrator-workers pattern emphasises **role-based task assignment**:
each ``WorkerAgent`` has a declared speciality, and the orchestrator
selects the best worker for each task.

References
----------
- Survey §4.4: Orchestrator-workers as a RAG workflow pattern.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.llm import LocalLLM
    from core.vector_store import FAISSVectorStore


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_PLAN_PROMPT = """You are an orchestrator responsible for breaking down a complex question into specific, executable sub-tasks.

Complex Question: {query}

Available worker specialties:
{worker_descriptions}

Create a detailed execution plan with 2–5 sub-tasks. Each sub-task must:
1. Be specific and actionable.
2. Have a single best-matched worker assigned.
3. Specify what information is needed from the worker.

Respond ONLY with a JSON array:
[
  {{
    "task_id": 1,
    "task_description": "<specific task description>",
    "assigned_worker": "<worker name from the available workers>",
    "expected_output": "<what this worker should produce>"
  }},
  ...
]"""

_COMPILE_PROMPT = """You are an expert report writer. Compile the results from multiple specialist workers into a single, coherent, comprehensive answer.

Original Question: {query}

Worker Results:
{worker_results_text}

Write a well-structured final answer that:
- Directly and completely addresses the original question
- Integrates insights from all workers
- Is logical, coherent, and professionally written
- Avoids repeating the same information

Final Compiled Answer:"""

_WORKER_EXECUTE_PROMPT = """You are a specialist worker with the following role: {role}

Task assigned to you: {task_description}

Query context: {query}

Retrieved information:
{context}

Complete your assigned task thoroughly and accurately based on your specialisation.

Task Output:"""


# ---------------------------------------------------------------------------
# WorkerAgent
# ---------------------------------------------------------------------------

@dataclass
class WorkerAgent:
    """
    A specialised agent that handles a specific type of sub-task.

    Attributes
    ----------
    name : str
        Unique identifier for this worker.
    speciality : str
        Human-readable description of what this worker is best at.
    llm : LocalLLM
        Language model used by this worker.
    vector_store : FAISSVectorStore
        Vector store used for retrieval in task execution.
    top_k : int
        Number of documents to retrieve per task (default 4).
    """

    name: str
    speciality: str
    llm: "LocalLLM"
    vector_store: "FAISSVectorStore"
    top_k: int = 4

    def _retrieve_context(self, query: str) -> str:
        """Retrieve relevant document chunks for the given query."""
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
        return "\n\n".join(chunks) if chunks else "No relevant documents found."

    def execute(self, task: dict, query: str) -> dict:
        """
        Execute an assigned task.

        Parameters
        ----------
        task : dict
            Task specification dict with keys: ``task_id``, ``task_description``,
            ``assigned_worker``, ``expected_output``.
        query : str
            The original user question (for retrieval context).

        Returns
        -------
        dict
            {
              "task_id"     : int|str,  # task identifier
              "worker_name" : str,      # name of this worker
              "task"        : str,      # the task description
              "result"      : str,      # the generated output
            }
        """
        task_description = task.get("task_description", "")
        # Use task description for retrieval (more specific than original query)
        context = self._retrieve_context(task_description)

        prompt = _WORKER_EXECUTE_PROMPT.format(
            role=self.speciality,
            task_description=task_description,
            query=query,
            context=context,
        )
        result = self.llm.generate(prompt).strip()

        return {
            "task_id": task.get("task_id", "N/A"),
            "worker_name": self.name,
            "task": task_description,
            "result": result,
        }


# ---------------------------------------------------------------------------
# OrchestratorAgent
# ---------------------------------------------------------------------------

class OrchestratorAgent:
    """
    Orchestrates a team of WorkerAgents to answer complex questions.

    The orchestrator dynamically generates a task plan, dispatches tasks to
    appropriate workers (with optional parallelism), and compiles the results.

    Parameters
    ----------
    llm : LocalLLM
        Language model for planning and compilation steps.
    vector_store : FAISSVectorStore
        Vector store for worker retrieval.
    max_workers_parallel : int, optional
        Maximum concurrent workers in the thread pool (default 3).
    """

    def __init__(
        self,
        llm: "LocalLLM",
        vector_store: "FAISSVectorStore",
        max_workers_parallel: int = 3,
    ) -> None:
        self.llm = llm
        self.vector_store = vector_store
        self.max_workers_parallel = max_workers_parallel
        self._worker_registry: dict[str, WorkerAgent] = {}

        # Register four default specialised workers
        self._register_default_workers()

    def _register_default_workers(self) -> None:
        """Register the four default specialised workers."""
        defaults = [
            WorkerAgent(
                name="retrieval_specialist",
                speciality=(
                    "Expert at finding and summarising relevant information from documents. "
                    "Best for: information gathering, document lookup, finding evidence."
                ),
                llm=self.llm,
                vector_store=self.vector_store,
            ),
            WorkerAgent(
                name="analysis_specialist",
                speciality=(
                    "Expert at deep analysis, reasoning, and drawing conclusions from data. "
                    "Best for: causal analysis, comparison, trend identification, reasoning chains."
                ),
                llm=self.llm,
                vector_store=self.vector_store,
            ),
            WorkerAgent(
                name="fact_checker",
                speciality=(
                    "Expert at verifying facts, checking consistency, and identifying inaccuracies. "
                    "Best for: fact verification, contradiction detection, accuracy checking."
                ),
                llm=self.llm,
                vector_store=self.vector_store,
            ),
            WorkerAgent(
                name="summarisation_specialist",
                speciality=(
                    "Expert at concisely summarising large amounts of information. "
                    "Best for: distilling key points, creating executive summaries, condensing long answers."
                ),
                llm=self.llm,
                vector_store=self.vector_store,
            ),
        ]
        for worker in defaults:
            self._worker_registry[worker.name] = worker

    def register_worker(self, worker: WorkerAgent) -> None:
        """
        Add a custom worker to the registry.

        Parameters
        ----------
        worker : WorkerAgent
            The worker to register. Overwrites any existing worker with the same name.
        """
        self._worker_registry[worker.name] = worker

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _plan_tasks(self, query: str) -> list[dict]:
        """
        Use the LLM to create a task list for the given query.

        The orchestrator is given the list of registered workers and their
        specialities, then outputs a JSON task plan assigning each task
        to the most suitable worker.

        Parameters
        ----------
        query : str
            The complex user question.

        Returns
        -------
        list[dict]
            A list of task dicts, each with: ``task_id``, ``task_description``,
            ``assigned_worker``, ``expected_output``.
        """
        worker_descriptions = "\n".join(
            f"- {name}: {worker.speciality}"
            for name, worker in self._worker_registry.items()
        )
        prompt = _PLAN_PROMPT.format(
            query=query, worker_descriptions=worker_descriptions
        )
        raw = self.llm.generate(prompt).strip()

        # Parse JSON array
        json_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if json_match:
            try:
                tasks = json.loads(json_match.group())
                if isinstance(tasks, list) and tasks:
                    return tasks[:5]  # cap at 5 tasks
            except json.JSONDecodeError:
                pass

        # Fallback: create two generic tasks
        return [
            {
                "task_id": 1,
                "task_description": f"Retrieve relevant information for: {query}",
                "assigned_worker": "retrieval_specialist",
                "expected_output": "Key information and facts",
            },
            {
                "task_id": 2,
                "task_description": f"Analyse and synthesise the information: {query}",
                "assigned_worker": "analysis_specialist",
                "expected_output": "Analytical conclusion",
            },
        ]

    def _dispatch(self, tasks: list[dict], query: str) -> list[dict]:
        """
        Assign and execute tasks concurrently across registered workers.

        If a task's ``assigned_worker`` is not registered, falls back to
        ``retrieval_specialist``.

        Parameters
        ----------
        tasks : list[dict]
            Task specifications from ``_plan_tasks``.
        query : str
            The original user query (passed to each worker for context retrieval).

        Returns
        -------
        list[dict]
            Ordered list of worker result dicts.
        """
        results: list[dict] = []

        with ThreadPoolExecutor(max_workers=self.max_workers_parallel) as executor:
            future_to_task = {}
            for task in tasks:
                worker_name = task.get("assigned_worker", "retrieval_specialist")
                worker = self._worker_registry.get(
                    worker_name,
                    self._worker_registry.get("retrieval_specialist"),
                )
                if worker is None:
                    # No workers at all — skip
                    continue
                future = executor.submit(worker.execute, task, query)
                future_to_task[future] = task

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    results.append(
                        {
                            "task_id": task.get("task_id"),
                            "worker_name": task.get("assigned_worker", "unknown"),
                            "task": task.get("task_description", ""),
                            "result": f"[Worker error: {exc}]",
                        }
                    )

        # Re-order results by task_id for deterministic output
        results.sort(key=lambda r: r.get("task_id", 0))
        return results

    def _compile_results(self, query: str, results: list[dict]) -> str:
        """
        Compile all worker results into a coherent final answer.

        Parameters
        ----------
        query : str
            The original user question.
        results : list[dict]
            Worker result dicts from ``_dispatch``.

        Returns
        -------
        str
            The compiled final answer.
        """
        worker_results_text = "\n\n".join(
            f"Worker: {r['worker_name']}\n"
            f"Task: {r['task']}\n"
            f"Output:\n{r['result']}"
            for r in results
        )
        prompt = _COMPILE_PROMPT.format(
            query=query, worker_results_text=worker_results_text
        )
        return self.llm.generate(prompt).strip()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def query(self, question: str) -> dict:
        """
        Run the full orchestrator-workers pipeline for a complex question.

        Workflow
        --------
        1. LLM generates a task plan assigning sub-tasks to workers.
        2. Workers execute their tasks concurrently (thread pool).
        3. Orchestrator compiles all results into a final answer.

        Parameters
        ----------
        question : str
            The complex user question.

        Returns
        -------
        dict
            {
              "task_plan"    : list[dict], # the generated execution plan
              "worker_results": list[dict],# per-task execution outputs
              "final_answer" : str,        # compiled final answer
              "latency"      : float,      # wall-clock time in seconds
            }
        """
        t_start = time.perf_counter()

        # Step 1: Plan
        task_plan = self._plan_tasks(question)

        # Step 2: Dispatch & execute
        worker_results = self._dispatch(task_plan, question)

        # Step 3: Compile
        final_answer = self._compile_results(question, worker_results)

        latency = time.perf_counter() - t_start

        return {
            "task_plan": task_plan,
            "worker_results": worker_results,
            "final_answer": final_answer,
            "latency": round(latency, 4),
        }
