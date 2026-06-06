"""
Reflection Pattern — §3.1 of 'Agentic RAG: A Survey' (arXiv:2501.09136)
=========================================================================
Implements the Self-Refine style reflection loop where an agent:
  1. Generates an initial answer from retrieved context.
  2. Critiques its own output, producing a structured feedback dict.
  3. Refines the answer based on the critique.
  4. Repeats until a quality threshold is met or the iteration cap is reached.

References
----------
- Madaan et al., "Self-Refine: Iterative Refinement with Self-Feedback", NeurIPS 2023.
- Survey §3.1: Reflection as an agentic design pattern for RAG systems.
"""

from __future__ import annotations

import json
import re
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.embeddings import Embedder
    from core.llm import LocalLLM
    from core.vector_store import FAISSVectorStore


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_INITIAL_ANSWER_PROMPT = """You are a knowledgeable assistant. Use the provided context to answer the question accurately and concisely.

Context:
{context}

Question: {query}

Answer:"""

_CRITIQUE_PROMPT = """You are a critical evaluator. Assess the following answer to a question and provide structured feedback.

Question: {query}

Answer:
{answer}

Evaluate the answer on these dimensions (score each 0–10):
- Relevance: Does it directly address the question?
- Completeness: Are all important aspects covered?
- Accuracy: Is the information factually correct and well-supported?
- Clarity: Is the answer well-structured and easy to understand?

Respond ONLY with valid JSON in this exact format:
{{
  "critique": "<detailed critique text>",
  "score": <average score 0.0–10.0>,
  "needs_improvement": <true|false>,
  "relevance": <0–10>,
  "completeness": <0–10>,
  "accuracy": <0–10>,
  "clarity": <0–10>
}}"""

_REFINE_PROMPT = """You are a knowledgeable assistant tasked with improving an answer based on critic feedback.

Original Question: {query}

Previous Answer:
{answer}

Critic Feedback:
{critique}

Please rewrite the answer, addressing all the issues raised in the feedback. Be thorough yet concise.

Improved Answer:"""


# ---------------------------------------------------------------------------
# ReflectionAgent
# ---------------------------------------------------------------------------

class ReflectionAgent:
    """
    Implements the Reflection (Self-Refine) agentic design pattern for RAG.

    The agent iteratively generates, critiques, and refines an answer until
    the critique score exceeds `quality_threshold` or `max_iterations` is
    reached.

    Parameters
    ----------
    llm : LocalLLM
        The language model used for generation, critique, and refinement.
    vector_store : FAISSVectorStore
        FAISS-backed vector store for document retrieval.
    max_iterations : int, optional
        Maximum number of refine cycles (default 3).
    quality_threshold : float, optional
        Score (0–10) above which the answer is considered good enough
        and the loop terminates early (default 8.0).
    top_k : int, optional
        Number of documents to retrieve from the vector store (default 5).
    """

    def __init__(
        self,
        llm: "LocalLLM",
        vector_store: "FAISSVectorStore",
        max_iterations: int = 3,
        quality_threshold: float = 8.0,
        top_k: int = 5,
    ) -> None:
        self.llm = llm
        self.vector_store = vector_store
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.top_k = top_k

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _retrieve_context(self, query: str) -> str:
        """
        Retrieve the top-k relevant documents from the vector store and
        concatenate them into a single context string.

        Parameters
        ----------
        query : str
            The user question used as the retrieval query.

        Returns
        -------
        str
            A newline-separated string of retrieved document chunks.
        """
        docs = self.vector_store.search(query, top_k=self.top_k)
        # `docs` is expected to be a list of strings or objects with `.text`
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

    def _generate_initial(self, query: str, context: str) -> str:
        """
        Generate the first-pass answer given the retrieved context.

        Parameters
        ----------
        query : str
            The original user question.
        context : str
            Concatenated retrieved document chunks.

        Returns
        -------
        str
            The raw generated answer text.
        """
        prompt = _INITIAL_ANSWER_PROMPT.format(context=context, query=query)
        return self.llm.generate(prompt).strip()

    def _critique(self, query: str, answer: str) -> dict:
        """
        Critique the given answer, returning a structured feedback dictionary.

        The LLM is asked to respond with JSON containing a `critique` text,
        a numeric `score`, a `needs_improvement` boolean, and per-criterion
        scores.  A fallback is applied if the LLM output cannot be parsed.

        Parameters
        ----------
        query : str
            The original user question.
        answer : str
            The answer to be critiqued.

        Returns
        -------
        dict
            Keys: critique (str), score (float), needs_improvement (bool),
                  relevance (int), completeness (int), accuracy (int),
                  clarity (int).
        """
        prompt = _CRITIQUE_PROMPT.format(query=query, answer=answer)
        raw = self.llm.generate(prompt).strip()

        # Attempt to extract JSON block from the response
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                # Ensure required keys are present with sensible defaults
                data.setdefault("critique", raw)
                data.setdefault("score", 5.0)
                data.setdefault("needs_improvement", True)
                data["score"] = float(data["score"])
                data["needs_improvement"] = bool(data["needs_improvement"])
                return data
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: treat entire response as critique text with neutral score
        return {
            "critique": raw,
            "score": 5.0,
            "needs_improvement": True,
            "relevance": 5,
            "completeness": 5,
            "accuracy": 5,
            "clarity": 5,
        }

    def _refine(self, query: str, answer: str, critique: str) -> str:
        """
        Produce an improved answer by incorporating the critique feedback.

        Parameters
        ----------
        query : str
            The original user question.
        answer : str
            The previous (unrefined) answer.
        critique : str
            Textual critique / feedback from `_critique`.

        Returns
        -------
        str
            The improved answer.
        """
        prompt = _REFINE_PROMPT.format(
            query=query, answer=answer, critique=critique
        )
        return self.llm.generate(prompt).strip()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def query(self, question: str) -> dict:
        """
        Run the full reflection loop for a given question.

        Workflow
        --------
        1. Retrieve relevant context from the vector store.
        2. Generate an initial answer.
        3. Critique the answer.
        4. If the score is below `quality_threshold` and there are
           remaining iterations, refine the answer and go to step 3.
        5. Return the final answer along with the full iteration history.

        Parameters
        ----------
        question : str
            The user question to answer.

        Returns
        -------
        dict
            {
              "final_answer"  : str,   # best answer after all iterations
              "iterations"    : int,   # number of cycles executed
              "history"       : list,  # per-iteration records
              "latency"       : float, # wall-clock time in seconds
            }

        Each history record contains::

            {
              "iteration": int,
              "answer"   : str,
              "critique" : str,
              "score"    : float,
            }
        """
        t_start = time.perf_counter()

        # Step 1: Retrieve context once (shared across all iterations)
        context = self._retrieve_context(question)

        # Step 2: Generate initial answer
        current_answer = self._generate_initial(question, context)

        history: list[dict] = []
        iterations = 0

        for iteration in range(1, self.max_iterations + 1):
            iterations = iteration

            # Step 3: Critique current answer
            critique_result = self._critique(question, current_answer)
            score = critique_result["score"]
            critique_text = critique_result.get("critique", "")

            history.append(
                {
                    "iteration": iteration,
                    "answer": current_answer,
                    "critique": critique_text,
                    "score": score,
                    "criteria": {
                        k: critique_result.get(k)
                        for k in ("relevance", "completeness", "accuracy", "clarity")
                    },
                }
            )

            # Early exit if quality threshold met
            if not critique_result.get("needs_improvement", True) or score >= self.quality_threshold:
                break

            # Step 4: Refine (skip on last iteration to save one LLM call)
            if iteration < self.max_iterations:
                current_answer = self._refine(question, current_answer, critique_text)

        latency = time.perf_counter() - t_start

        return {
            "final_answer": current_answer,
            "iterations": iterations,
            "history": history,
            "latency": round(latency, 4),
        }
