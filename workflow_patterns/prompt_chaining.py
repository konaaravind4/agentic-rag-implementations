"""
Prompt Chaining — §4.1 of 'Agentic RAG: A Survey' (arXiv:2501.09136)
=====================================================================
Implements a sequential prompt-chaining workflow where the output of each
step becomes the input to the next, enabling complex multi-stage reasoning
and processing pipelines.

The demo chain provided mirrors the retrieval-augmented answer generation
pipeline:
  1. query_clarification   — rewrite/clarify the user's question
  2. document_retrieval    — identify key retrieval terms
  3. answer_generation     — generate an answer using retrieved context
  4. citation_formatting   — format the answer with inline citation hints

References
----------
- Wu et al., "PromptChainer: Chaining Large Language Model Prompts", CHI 2022.
- Survey §4.1: Prompt chaining as a RAG workflow pattern.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.llm import LocalLLM
    from core.vector_store import FAISSVectorStore


# ---------------------------------------------------------------------------
# Demo chain step prompts
# ---------------------------------------------------------------------------

_CLARIFICATION_TEMPLATE = """You are a query understanding specialist. Your job is to rewrite and clarify the user's question to make it more precise and answerable.

Original Query: {input}

Rewrite the query to:
- Make it more specific and unambiguous
- Expand any acronyms or vague references
- Preserve the original intent

Clarified Query:"""

_RETRIEVAL_TERMS_TEMPLATE = """You are a search expert. Given a clarified query, extract the most important keywords and search terms for document retrieval.

Clarified Query: {input}

List the 5–8 most important search terms and concepts (comma-separated), ordered by importance:"""

_ANSWER_GENERATION_TEMPLATE = """You are a knowledgeable assistant. Given the search terms and retrieved context below, generate a comprehensive answer to the original query.

Search Terms and Context: {input}

Generate a well-structured, factual answer that directly addresses the query. Include specific details where available.

Answer:"""

_CITATION_FORMAT_TEMPLATE = """You are a technical writer. Take the following answer and format it professionally with source attribution markers.

Answer to Format: {input}

Reformat the answer to:
- Add [Source N] citation markers where specific facts are stated
- Add a brief "References" section at the end with placeholder citations
- Maintain the factual content while improving presentation

Formatted Answer with Citations:"""


# ---------------------------------------------------------------------------
# PromptChaining
# ---------------------------------------------------------------------------

class PromptChaining:
    """
    Implements the Prompt Chaining workflow pattern for RAG (§4.1).

    Steps are added sequentially via ``add_step``; each step receives the
    previous step's output as ``{input}`` in its template, and produces a
    new output which is passed forward.

    Parameters
    ----------
    llm : LocalLLM
        Language model used for all chain steps.
    vector_store : FAISSVectorStore, optional
        If provided, used to inject retrieved context into designated steps.
    top_k : int, optional
        Number of documents to retrieve (default 4).
    """

    def __init__(
        self,
        llm: "LocalLLM",
        vector_store: "FAISSVectorStore | None" = None,
        top_k: int = 4,
    ) -> None:
        self.llm = llm
        self.vector_store = vector_store
        self.top_k = top_k
        self._steps: list[dict] = []

    # ------------------------------------------------------------------
    # Step management
    # ------------------------------------------------------------------

    def add_step(self, name: str, prompt_template: str) -> "PromptChaining":
        """
        Append a step to the chain.

        Parameters
        ----------
        name : str
            A human-readable label for this step (used in output dict).
        prompt_template : str
            A prompt string containing exactly one ``{input}`` placeholder
            that will be replaced with the output of the preceding step
            (or the initial input for the first step).

        Returns
        -------
        PromptChaining
            Returns ``self`` to support method chaining (fluent API).
        """
        if "{input}" not in prompt_template:
            raise ValueError(
                f"Step '{name}': prompt_template must contain a '{{input}}' placeholder."
            )
        self._steps.append({"name": name, "template": prompt_template})
        return self

    def clear_steps(self) -> None:
        """Remove all steps from the chain."""
        self._steps.clear()

    # ------------------------------------------------------------------
    # Retrieval helper
    # ------------------------------------------------------------------

    def _retrieve_context(self, query: str) -> str:
        """Retrieve documents from the vector store (if configured)."""
        if self.vector_store is None:
            return ""
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

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, initial_input: str) -> dict:
        """
        Execute all steps sequentially, threading outputs as inputs.

        The first step receives ``initial_input``; every subsequent step
        receives the output of the previous step.

        Special behaviour: if the step is named ``'answer_generation'``
        **and** a ``vector_store`` is configured, retrieved context is
        appended to the input before calling the LLM.

        Parameters
        ----------
        initial_input : str
            The starting input fed into the first step.

        Returns
        -------
        dict
            {
              "steps_output"  : dict[str, str],  # per-step {name: output}
              "final_output"  : str,             # output of the last step
              "latency"       : float,           # wall-clock time in seconds
              "step_latencies": dict[str, float],# per-step timing
            }

        Raises
        ------
        RuntimeError
            If the chain has no steps added.
        """
        if not self._steps:
            raise RuntimeError(
                "No steps in the chain. Add steps with add_step() before calling run()."
            )

        t_start = time.perf_counter()
        current_input = initial_input
        steps_output: dict[str, str] = {}
        step_latencies: dict[str, float] = {}

        for step in self._steps:
            step_name = step["name"]
            template = step["template"]

            # Inject retrieved context for the answer generation step
            effective_input = current_input
            if step_name == "answer_generation" and self.vector_store is not None:
                context = self._retrieve_context(initial_input)
                if context:
                    effective_input = (
                        f"{current_input}\n\nRetrieved Context:\n{context}"
                    )

            prompt = template.format(input=effective_input)

            step_t_start = time.perf_counter()
            output = self.llm.generate(prompt).strip()
            step_latencies[step_name] = round(time.perf_counter() - step_t_start, 4)

            steps_output[step_name] = output
            current_input = output  # thread output to next step

        latency = time.perf_counter() - t_start

        return {
            "steps_output": steps_output,
            "final_output": current_input,
            "latency": round(latency, 4),
            "step_latencies": step_latencies,
        }

    # ------------------------------------------------------------------
    # Demo chain factory
    # ------------------------------------------------------------------

    @classmethod
    def build_rag_demo_chain(
        cls,
        llm: "LocalLLM",
        vector_store: "FAISSVectorStore | None" = None,
        top_k: int = 4,
    ) -> "PromptChaining":
        """
        Build the four-step RAG demo chain described in §4.1.

        Steps:
        1. ``query_clarification``  — sharpen and clarify the query.
        2. ``document_retrieval``   — extract retrieval terms.
        3. ``answer_generation``    — generate an answer (+ optional RAG).
        4. ``citation_formatting``  — add professional citations.

        Parameters
        ----------
        llm : LocalLLM
            Language model for all steps.
        vector_store : FAISSVectorStore, optional
            If provided, context is retrieved for the answer_generation step.
        top_k : int, optional
            Documents to retrieve (default 4).

        Returns
        -------
        PromptChaining
            A fully configured ``PromptChaining`` instance.
        """
        chain = cls(llm=llm, vector_store=vector_store, top_k=top_k)
        chain.add_step("query_clarification", _CLARIFICATION_TEMPLATE)
        chain.add_step("document_retrieval", _RETRIEVAL_TERMS_TEMPLATE)
        chain.add_step("answer_generation", _ANSWER_GENERATION_TEMPLATE)
        chain.add_step("citation_formatting", _CITATION_FORMAT_TEMPLATE)
        return chain
