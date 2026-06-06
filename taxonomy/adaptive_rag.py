"""
taxonomy/adaptive_rag.py
========================
Adaptive Agentic RAG — §5.5 of 'Agentic RAG: A Survey' (arXiv:2501.09136)
--------------------------------------------------------------------------
Adaptive RAG routes each query to one of three processing paths based on
estimated query complexity:

  * no_retrieval   — Simple factoid; answer directly from LLM parametric knowledge.
  * single_step    — Single-entity question; one vector store lookup + generation.
  * multi_step     — Complex / multi-part question; iterative retrieval + reasoning.

The classifier uses lightweight heuristics first (word count, WH-words,
entity density) and falls back to an LLM prompt when the heuristics are
ambiguous.

References
----------
- Jeong et al., "Adaptive-RAG", 2024.
- Survey §5.5.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional

from core.llm import LocalLLM
from core.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMPLEXITY_CLASSES = ("no_retrieval", "single_step", "multi_step")

# Heuristic thresholds
_SHORT_QUERY_MAX_TOKENS = 6         # ≤ 6 tokens → likely factoid
_MULTI_STEP_MIN_TOKENS  = 12        # ≥ 12 tokens → likely complex
_MULTI_STEP_KEYWORDS = frozenset([
    "why", "how", "compare", "difference", "explain",
    "relationship", "impact", "effect", "cause", "because",
    "however", "although", "despite", "versus", "vs",
    "contrast", "similar", "different", "both", "between",
])
_FACTOID_PATTERNS = re.compile(
    r"^(what is|who is|when was|where is|which|define|what are|how many|"
    r"what does .* stand for)\b",
    re.IGNORECASE,
)

# LLM prompt for ambiguous cases
_CLASSIFY_PROMPT = """\
You are a query complexity classifier for a RAG system.
Classify the following question into EXACTLY ONE of these three categories:

1. no_retrieval   — Very simple factoid; the answer is a common fact (e.g. definitions,
                    dates, short answers).
2. single_step    — Requires retrieving information about a single concept/entity.
3. multi_step     — Complex; requires multi-hop reasoning, comparisons, or
                    synthesis across multiple topics.

Question: "{query}"

Respond with ONLY one of these words: no_retrieval, single_step, multi_step
Answer:"""

# Answer generation prompt
_ANSWER_PROMPT = """\
Answer the following question concisely and accurately.

{context_section}Question: {query}

Answer:"""


# ---------------------------------------------------------------------------
# ComplexityClassifier
# ---------------------------------------------------------------------------

class ComplexityClassifier:
    """
    Classifies query complexity using a two-stage approach:

    1. **Heuristic stage** — rules based on token count, WH-words, and
       factoid-pattern matching (zero LLM calls, fast).
    2. **LLM stage** — used only when heuristics are ambiguous (i.e. the
       query falls in a "grey zone" between complexity classes).

    Parameters
    ----------
    llm : LocalLLM
        Language model used for ambiguous classification cases.
    use_llm_fallback : bool
        Whether to invoke the LLM for grey-zone queries (default ``True``).
        Set to ``False`` for tests / offline mode.
    """

    def __init__(
        self,
        llm: LocalLLM,
        use_llm_fallback: bool = True,
    ) -> None:
        self.llm = llm
        self.use_llm_fallback = use_llm_fallback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, query: str) -> str:
        """
        Determine the complexity class of ``query``.

        Parameters
        ----------
        query : str
            The raw user question.

        Returns
        -------
        str
            One of ``'no_retrieval'``, ``'single_step'``, ``'multi_step'``.
        """
        # Stage 1: heuristics
        heuristic_result = self._heuristic_classify(query)
        if heuristic_result is not None:
            logger.debug(
                "[Classifier] Heuristic → %s | query=%r", heuristic_result, query
            )
            return heuristic_result

        # Stage 2: LLM fallback
        if self.use_llm_fallback:
            llm_result = self._llm_classify(query)
            logger.debug(
                "[Classifier] LLM → %s | query=%r", llm_result, query
            )
            return llm_result

        # Default: play it safe with single_step retrieval
        return "single_step"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _heuristic_classify(self, query: str) -> Optional[str]:
        """
        Apply rule-based heuristics to classify the query.

        Returns ``None`` when the heuristics are inconclusive so the
        LLM fallback can take over.
        """
        tokens = query.strip().lower().split()
        n_tokens = len(tokens)
        token_set = set(tokens)

        # Rule 1: Very short + factoid pattern → no_retrieval
        if n_tokens <= _SHORT_QUERY_MAX_TOKENS:
            if _FACTOID_PATTERNS.match(query.strip()):
                return "no_retrieval"
            # Even without factoid pattern, very short → no_retrieval heuristic
            if n_tokens <= 4:
                return "no_retrieval"

        # Rule 2: Multi-step keywords present OR long query → multi_step
        multi_kw_hits = token_set & _MULTI_STEP_KEYWORDS
        if multi_kw_hits or n_tokens >= _MULTI_STEP_MIN_TOKENS:
            # Only commit to multi_step if at least one signal is strong
            if len(multi_kw_hits) >= 2 or n_tokens >= _MULTI_STEP_MIN_TOKENS:
                return "multi_step"

        # Rule 3: Medium length, no multi-step keywords → single_step
        if _SHORT_QUERY_MAX_TOKENS < n_tokens < _MULTI_STEP_MIN_TOKENS:
            return "single_step"

        # Ambiguous — let the LLM decide
        return None

    def _llm_classify(self, query: str) -> str:
        """
        Use the LLM to classify a query as a fallback.

        The LLM is asked to output exactly one of the three class names.
        Any unrecognised output defaults to ``'single_step'``.
        """
        prompt = _CLASSIFY_PROMPT.format(query=query)
        try:
            raw = self.llm.generate(prompt, max_new_tokens=16, temperature=0.0).strip().lower()
            # Extract first recognised class name from output
            for cls in COMPLEXITY_CLASSES:
                if cls in raw:
                    return cls
        except Exception as exc:
            logger.warning("LLM classification failed: %s", exc)

        return "single_step"  # safe default


# ---------------------------------------------------------------------------
# AdaptiveRAG
# ---------------------------------------------------------------------------

class AdaptiveRAG:
    """
    Adaptive Agentic RAG — §5.5.

    Routes each incoming question to the appropriate processing path based
    on query complexity.  The routing decision is made by
    :class:`ComplexityClassifier`.

    Parameters
    ----------
    llm : LocalLLM
        Language model for generation and classification fallback.
    vector_store : FAISSVectorStore
        Indexed document store for retrieval.
    top_k : int
        Number of documents to retrieve per step (default 3).
    max_multi_step_rounds : int
        Maximum iterative retrieval rounds for multi-step queries (default 3).
    use_llm_classifier : bool
        Whether the classifier may call the LLM for ambiguous cases.
    """

    def __init__(
        self,
        llm: LocalLLM,
        vector_store: FAISSVectorStore,
        top_k: int = 3,
        max_multi_step_rounds: int = 3,
        use_llm_classifier: bool = True,
    ) -> None:
        self.llm = llm
        self.vector_store = vector_store
        self.top_k = top_k
        self.max_multi_step_rounds = max_multi_step_rounds
        self.classifier = ComplexityClassifier(llm, use_llm_fallback=use_llm_classifier)

    # ------------------------------------------------------------------
    # Path implementations
    # ------------------------------------------------------------------

    def _no_retrieval_path(self, query: str) -> Dict[str, Any]:
        """
        Direct LLM answer with no document retrieval.

        Appropriate for simple factoid questions where the LLM's parametric
        knowledge is sufficient.

        Parameters
        ----------
        query : str
            User question.

        Returns
        -------
        dict
            Keys: ``answer``, ``retrieved_docs``, ``rounds``.
        """
        logger.info("[Path] no_retrieval | query=%r", query)
        prompt = _ANSWER_PROMPT.format(context_section="", query=query)
        answer = self.llm.generate(prompt).strip()
        return {
            "answer": answer,
            "retrieved_docs": [],
            "rounds": 0,
        }

    def _single_step_path(self, query: str) -> Dict[str, Any]:
        """
        Single FAISS lookup followed by conditioned generation.

        Parameters
        ----------
        query : str
            User question.

        Returns
        -------
        dict
            Keys: ``answer``, ``retrieved_docs``, ``rounds``.
        """
        logger.info("[Path] single_step | query=%r", query)
        docs = self._retrieve(query)
        context = self._format_context(docs)
        answer = self._generate_answer(query, context)
        return {
            "answer": answer,
            "retrieved_docs": docs,
            "rounds": 1,
        }

    def _multi_step_path(self, query: str) -> Dict[str, Any]:
        """
        Iterative retrieval and reasoning (2–3 rounds).

        Each round:
        1. Retrieves documents for the current sub-query.
        2. Uses the LLM to extract a reasoning step and identify whether
           more retrieval is needed (sub-question decomposition).
        3. Refines the query for the next round based on the intermediate answer.

        Parameters
        ----------
        query : str
            The original complex question.

        Returns
        -------
        dict
            Keys: ``answer``, ``retrieved_docs``, ``rounds``,
            ``reasoning_trace`` (list of per-round dicts).
        """
        logger.info("[Path] multi_step | query=%r", query)

        all_docs: List[Dict[str, Any]] = []
        reasoning_trace: List[Dict[str, Any]] = []
        current_query = query
        accumulated_context = ""

        for round_idx in range(1, self.max_multi_step_rounds + 1):
            logger.debug("[multi_step] Round %d | sub-query=%r", round_idx, current_query)

            # Retrieve for current sub-query
            docs = self._retrieve(current_query)
            all_docs.extend(docs)
            round_context = self._format_context(docs)
            accumulated_context += f"\n\n[Round {round_idx} context]\n{round_context}"

            # Generate intermediate reasoning + potential sub-query
            intermediate_prompt = (
                f"You are answering a complex question step by step.\n\n"
                f"Original question: {query}\n\n"
                f"Accumulated context so far:\n{accumulated_context}\n\n"
                f"Step {round_idx}: Provide a partial answer and, if more information "
                f"is needed, write a short follow-up search query prefixed with "
                f"'SEARCH: '. If the question is fully answered, write 'DONE'.\n\n"
                f"Partial answer / next step:"
            )
            step_output = self.llm.generate(intermediate_prompt, max_new_tokens=128).strip()

            reasoning_trace.append({
                "round": round_idx,
                "sub_query": current_query,
                "docs_retrieved": len(docs),
                "step_output": step_output,
            })

            # Check for early termination or new sub-query
            if "DONE" in step_output.upper():
                logger.debug("[multi_step] Agent signals DONE at round %d.", round_idx)
                break

            search_match = re.search(r"SEARCH:\s*(.+)", step_output, re.IGNORECASE)
            if search_match:
                current_query = search_match.group(1).strip()
            else:
                # No explicit sub-query → use original query refined with what we know
                current_query = query  # fallback

        # Final synthesis from all accumulated context
        final_context = self._format_context(all_docs)
        final_answer = self._generate_answer(query, final_context)

        return {
            "answer": final_answer,
            "retrieved_docs": all_docs,
            "rounds": len(reasoning_trace),
            "reasoning_trace": reasoning_trace,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve top-k documents from the vector store."""
        try:
            results = self.vector_store.search(query, top_k=self.top_k)
        except Exception:
            results = self.vector_store.similarity_search(query, k=self.top_k)
        return results

    @staticmethod
    def _format_context(docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into a context string."""
        if not docs:
            return "No relevant documents found."
        parts = []
        for i, doc in enumerate(docs, 1):
            text = doc.get("text", doc.get("content", str(doc)))
            parts.append(f"[Doc {i}] {text}")
        return "\n\n".join(parts)

    def _generate_answer(self, query: str, context: str) -> str:
        """Generate a final answer conditioned on context."""
        context_section = f"Context:\n{context}\n\n"
        prompt = _ANSWER_PROMPT.format(
            context_section=context_section, query=query
        )
        return self.llm.generate(prompt).strip()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def query(self, question: str) -> Dict[str, Any]:
        """
        Classify and answer a question using the appropriate RAG path.

        Parameters
        ----------
        question : str
            The user's natural-language question.

        Returns
        -------
        dict
            {
              ``complexity``   : str   — classified complexity class,
              ``path_taken``   : str   — same as complexity (explicit alias),
              ``answer``       : str   — the generated answer,
              ``latency``      : float — end-to-end wall-clock time in seconds,
              ``retrieved_docs`` : list — documents retrieved (empty for no_retrieval),
              ``rounds``       : int  — retrieval rounds executed,
              ``reasoning_trace`` : list — per-round trace (multi_step only),
            }
        """
        logger.info("=== AdaptiveRAG.query | question=%r ===", question)
        t_start = time.perf_counter()

        # Step 1: Classify
        complexity = self.classifier.classify(question)
        logger.info("[AdaptiveRAG] Complexity=%r", complexity)

        # Step 2: Route to appropriate path
        dispatch = {
            "no_retrieval": self._no_retrieval_path,
            "single_step":  self._single_step_path,
            "multi_step":   self._multi_step_path,
        }
        path_fn = dispatch.get(complexity, self._single_step_path)
        path_result = path_fn(question)

        latency = time.perf_counter() - t_start

        return {
            "complexity":     complexity,
            "path_taken":     complexity,
            "answer":         path_result["answer"],
            "latency":        round(latency, 4),
            "retrieved_docs": path_result.get("retrieved_docs", []),
            "rounds":         path_result.get("rounds", 0),
            "reasoning_trace": path_result.get("reasoning_trace", []),
        }


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("AdaptiveRAG — smoke test (heuristic classifier only)")
    print("=" * 60)

    class _MockLLM:
        model_name = "mock"
        def generate(self, prompt, **_):
            return "This is a mock answer."

    class _MockStore:
        def search(self, query, top_k=3):
            return [{"text": f"Doc about {query}", "score": 0.9}]
        def similarity_search(self, query, k=3):
            return self.search(query, top_k=k)

    rag = AdaptiveRAG(_MockLLM(), _MockStore(), use_llm_classifier=False)

    test_queries = [
        "What is FAISS?",
        "How does diabetes affect metabolism?",
        "Compare BM25 and dense retrieval methods for large-scale document search.",
    ]
    for q in test_queries:
        result = rag.query(q)
        print(f"Q: {q}")
        print(f"  Complexity : {result['complexity']}")
        print(f"  Rounds     : {result['rounds']}")
        print(f"  Latency    : {result['latency']:.4f}s")
        print(f"  Answer     : {result['answer'][:80]}...")
        print()
