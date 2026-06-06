"""
Evaluator-Optimizer — §4.5 of 'Agentic RAG: A Survey' (arXiv:2501.09136)
=========================================================================
Implements a generate-evaluate-refine loop where an LLM-based evaluator
scores the generated answer against four quality criteria and the generator
iteratively improves until a target score is achieved or the iteration
budget is exhausted.

Evaluation criteria:
- **Relevance**    — Does the answer directly address the question?
- **Completeness** — Are all important aspects of the question covered?
- **Accuracy**     — Are the stated facts correct and well-supported?
- **Clarity**      — Is the answer logically structured and easy to follow?

Each criterion is scored 0–10 and the overall score is the unweighted mean.

References
----------
- Survey §4.5: Evaluator-optimizer as a RAG workflow pattern.
- Madaan et al., "Self-Refine", NeurIPS 2023 (criterion-based variant).
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
# Evaluation criteria metadata
# ---------------------------------------------------------------------------

EVALUATION_CRITERIA = {
    "relevance": "How directly and precisely the answer addresses the specific question asked.",
    "completeness": "Whether all important aspects and sub-points of the question are covered.",
    "accuracy": "Whether the stated facts are correct, well-supported, and free from errors.",
    "clarity": "Whether the answer is well-structured, logically ordered, and easy to understand.",
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_GENERATE_PROMPT = """You are a knowledgeable assistant. Generate a comprehensive answer to the question using the provided context.

Context:
{context}

Question: {query}

{feedback_section}

Write a high-quality, well-structured answer that is accurate, complete, and clear:

Answer:"""

_EVALUATE_PROMPT = """You are an expert evaluator assessing the quality of an answer.

Question: {query}

Answer to Evaluate:
{answer}

Score the answer on each of the following criteria (0–10):
- Relevance (0–10): {relevance_desc}
- Completeness (0–10): {completeness_desc}
- Accuracy (0–10): {accuracy_desc}
- Clarity (0–10): {clarity_desc}

For each criterion, provide:
1. A score (0–10)
2. A brief justification
3. Specific suggestions for improvement (if score < 8)

Respond ONLY with valid JSON:
{{
  "relevance": {{
    "score": <0–10>,
    "justification": "<brief text>",
    "suggestions": "<improvement suggestions or 'None needed'>"
  }},
  "completeness": {{
    "score": <0–10>,
    "justification": "<brief text>",
    "suggestions": "<improvement suggestions or 'None needed'>"
  }},
  "accuracy": {{
    "score": <0–10>,
    "justification": "<brief text>",
    "suggestions": "<improvement suggestions or 'None needed'>"
  }},
  "clarity": {{
    "score": <0–10>,
    "justification": "<brief text>",
    "suggestions": "<improvement suggestions or 'None needed'>"
  }},
  "overall_feedback": "<overall assessment and primary improvement recommendation>"
}}"""


# ---------------------------------------------------------------------------
# EvaluatorOptimizer
# ---------------------------------------------------------------------------

class EvaluatorOptimizer:
    """
    Implements the Evaluator-Optimizer workflow pattern for RAG (§4.5).

    The system runs a generate → evaluate → generate loop, using
    structured LLM-based evaluation to guide answer improvement.

    Parameters
    ----------
    llm : LocalLLM
        Language model for both generation and evaluation.
    vector_store : FAISSVectorStore
        Vector store for document retrieval.
    quality_threshold : float, optional
        Overall score (0–10) at which the loop terminates early (default 8.0).
    top_k : int, optional
        Number of documents to retrieve per query (default 5).
    """

    def __init__(
        self,
        llm: "LocalLLM",
        vector_store: "FAISSVectorStore",
        quality_threshold: float = 8.0,
        top_k: int = 5,
    ) -> None:
        self.llm = llm
        self.vector_store = vector_store
        self.quality_threshold = quality_threshold
        self.top_k = top_k

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _retrieve_context(self, query: str) -> str:
        """Retrieve top-k document chunks for the query."""
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

    def _generate(self, query: str, context: str, feedback: str = "") -> str:
        """
        Generate (or regenerate) an answer, optionally incorporating prior feedback.

        Parameters
        ----------
        query : str
            The user question.
        context : str
            Retrieved document context.
        feedback : str, optional
            Evaluation feedback from the previous iteration.  An empty string
            means this is the first generation.

        Returns
        -------
        str
            The generated answer text.
        """
        if feedback:
            feedback_section = (
                "IMPORTANT — Previous evaluation feedback to incorporate:\n"
                f"{feedback}\n\n"
                "Please address ALL of the above issues in your new answer."
            )
        else:
            feedback_section = ""

        prompt = _GENERATE_PROMPT.format(
            context=context,
            query=query,
            feedback_section=feedback_section,
        )
        return self.llm.generate(prompt).strip()

    def _evaluate(self, query: str, answer: str) -> dict:
        """
        Evaluate an answer against all four quality criteria.

        The evaluator LLM is prompted for a structured JSON response with
        per-criterion scores, justifications, and improvement suggestions.
        A robust fallback is applied if JSON parsing fails.

        Parameters
        ----------
        query : str
            The original user question.
        answer : str
            The answer to evaluate.

        Returns
        -------
        dict
            {
              "score"          : float,        # overall mean score (0–10)
              "feedback"       : str,          # compiled improvement feedback
              "criteria_scores": dict[str, dict],  # per-criterion details
              "passes_threshold": bool,        # score >= quality_threshold
            }

        Each ``criteria_scores`` entry::

            {
              "score"       : int,
              "justification": str,
              "suggestions" : str,
            }
        """
        prompt = _EVALUATE_PROMPT.format(
            query=query,
            answer=answer,
            **{f"{k}_desc": v for k, v in EVALUATION_CRITERIA.items()},
        )
        raw = self.llm.generate(prompt).strip()

        # Parse JSON response
        criteria_scores: dict[str, dict] = {}
        overall_feedback = ""

        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                for criterion in EVALUATION_CRITERIA:
                    if criterion in data:
                        entry = data[criterion]
                        criteria_scores[criterion] = {
                            "score": float(entry.get("score", 5)),
                            "justification": entry.get("justification", ""),
                            "suggestions": entry.get("suggestions", ""),
                        }
                overall_feedback = data.get("overall_feedback", "")
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        # Fallback: assign neutral scores if parsing failed
        if not criteria_scores:
            for criterion in EVALUATION_CRITERIA:
                criteria_scores[criterion] = {
                    "score": 5.0,
                    "justification": "Parsing failed; neutral score assigned.",
                    "suggestions": "Please review and improve the answer.",
                }
            overall_feedback = raw  # use raw text as feedback

        # Calculate overall score as mean of criteria scores
        score_values = [c["score"] for c in criteria_scores.values()]
        overall_score = sum(score_values) / len(score_values) if score_values else 5.0

        # Compile actionable feedback from suggestions
        feedback_parts = []
        for criterion, detail in criteria_scores.items():
            suggestions = detail.get("suggestions", "")
            if suggestions and suggestions.lower() not in ("none needed", "n/a", ""):
                feedback_parts.append(f"[{criterion.capitalize()}] {suggestions}")
        if overall_feedback:
            feedback_parts.append(f"[Overall] {overall_feedback}")
        compiled_feedback = "\n".join(feedback_parts) if feedback_parts else "Answer is satisfactory."

        return {
            "score": round(overall_score, 2),
            "feedback": compiled_feedback,
            "criteria_scores": criteria_scores,
            "passes_threshold": overall_score >= self.quality_threshold,
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def query(self, question: str, max_iterations: int = 3) -> dict:
        """
        Run the full evaluator-optimizer loop for a question.

        Workflow
        --------
        1. Retrieve context once (shared across all iterations).
        2. Generate an initial answer.
        3. Evaluate the answer; if score ≥ ``quality_threshold``, stop.
        4. Use the evaluation feedback to regenerate a better answer.
        5. Repeat steps 3–4 up to ``max_iterations`` times.

        Parameters
        ----------
        question : str
            The user question to answer.
        max_iterations : int, optional
            Maximum number of generate-evaluate cycles (default 3).

        Returns
        -------
        dict
            {
              "iterations"    : int,        # cycles completed
              "final_answer"  : str,        # best answer produced
              "final_score"   : float,      # final overall quality score
              "history"       : list[dict], # per-iteration records
              "latency"       : float,      # wall-clock time in seconds
            }

        Each history record::

            {
              "iteration"      : int,
              "answer"         : str,
              "score"          : float,
              "feedback"       : str,
              "criteria_scores": dict,
            }
        """
        t_start = time.perf_counter()

        # Step 1: Retrieve context (once)
        context = self._retrieve_context(question)

        history: list[dict] = []
        current_feedback = ""
        current_answer = ""
        current_score = 0.0
        iterations_run = 0

        for iteration in range(1, max_iterations + 1):
            iterations_run = iteration

            # Step 2/4: Generate (or regenerate with feedback)
            current_answer = self._generate(question, context, current_feedback)

            # Step 3: Evaluate
            evaluation = self._evaluate(question, current_answer)
            current_score = evaluation["score"]
            current_feedback = evaluation["feedback"]

            history.append(
                {
                    "iteration": iteration,
                    "answer": current_answer,
                    "score": current_score,
                    "feedback": current_feedback,
                    "criteria_scores": evaluation["criteria_scores"],
                }
            )

            # Early exit if quality threshold is met
            if evaluation["passes_threshold"]:
                break

        latency = time.perf_counter() - t_start

        return {
            "iterations": iterations_run,
            "final_answer": current_answer,
            "final_score": current_score,
            "history": history,
            "latency": round(latency, 4),
        }
