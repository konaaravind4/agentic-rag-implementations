"""
Evaluation metrics for Agentic RAG systems.

Implements faithfulness, relevance, F1, exact match, and BLEU scoring,
plus a RAGEvaluator class for comparing systems.

Paper: arXiv:2501.09136, §11 — Benchmarks and Datasets
"""

import re
import math
import numpy as np
import pandas as pd
from collections import Counter
from typing import Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Standalone metric functions
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list:
    """Lowercase, remove punctuation, split into tokens."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


def faithfulness_score(answer: str, context: str) -> float:
    """
    Measure how grounded the answer is in the retrieved context.

    Uses token-level overlap: proportion of answer tokens that appear
    in the context.

    Args:
        answer: Generated answer string.
        context: Concatenated retrieved document text.

    Returns:
        Float in [0, 1]. Higher = more faithful.
    """
    answer_tokens = set(_tokenize(answer))
    context_tokens = set(_tokenize(context))
    if not answer_tokens:
        return 0.0
    overlap = answer_tokens & context_tokens
    return round(len(overlap) / len(answer_tokens), 4)


def relevance_score(query: str, answer: str, embedder=None) -> float:
    """
    Measure semantic similarity between the query and answer.

    Uses cosine similarity of sentence embeddings if an embedder is provided,
    otherwise falls back to token Jaccard similarity.

    Args:
        query: The user query.
        answer: The generated answer.
        embedder: Optional Embedder instance for dense similarity.

    Returns:
        Float in [0, 1]. Higher = more relevant.
    """
    if embedder is not None:
        try:
            q_emb = embedder.embed_query(query)
            a_emb = embedder.embed_query(answer)
            # Cosine similarity
            dot = float(np.dot(q_emb, a_emb))
            norm = float(np.linalg.norm(q_emb) * np.linalg.norm(a_emb))
            if norm == 0:
                return 0.0
            return round(max(0.0, min(1.0, dot / norm)), 4)
        except Exception:
            pass
    # Fallback: Jaccard
    q_tokens = set(_tokenize(query))
    a_tokens = set(_tokenize(answer))
    if not q_tokens and not a_tokens:
        return 1.0
    union = q_tokens | a_tokens
    if not union:
        return 0.0
    return round(len(q_tokens & a_tokens) / len(union), 4)


def f1_score(predicted: str, ground_truth: str) -> float:
    """
    Token-level F1 score between predicted and ground-truth strings.

    Args:
        predicted: Model-generated answer.
        ground_truth: Reference answer.

    Returns:
        Float in [0, 1].
    """
    pred_tokens = _tokenize(predicted)
    truth_tokens = _tokenize(ground_truth)
    if not pred_tokens or not truth_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    truth_counter = Counter(truth_tokens)

    common = sum((pred_counter & truth_counter).values())
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


def exact_match(predicted: str, ground_truth: str) -> float:
    """
    Exact string match after normalisation (lowercase, strip whitespace).

    Returns:
        1.0 if exact match, 0.0 otherwise.
    """
    pred_norm = " ".join(_tokenize(predicted))
    truth_norm = " ".join(_tokenize(ground_truth))
    return 1.0 if pred_norm == truth_norm else 0.0


def bleu_score(predicted: str, reference: str, max_n: int = 1) -> float:
    """
    Simplified n-gram BLEU score (up to max_n=1 by default).

    Args:
        predicted: Hypothesis string.
        reference: Reference string.
        max_n: Maximum n-gram order (1 = unigram BLEU).

    Returns:
        Float in [0, 1].
    """
    pred_tokens = _tokenize(predicted)
    ref_tokens = _tokenize(reference)

    if not pred_tokens:
        return 0.0

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1)))

    scores = []
    for n in range(1, max_n + 1):
        pred_ngrams = Counter(
            tuple(pred_tokens[i: i + n]) for i in range(len(pred_tokens) - n + 1)
        )
        ref_ngrams = Counter(
            tuple(ref_tokens[i: i + n]) for i in range(len(ref_tokens) - n + 1)
        )
        clipped = sum((pred_ngrams & ref_ngrams).values())
        total = sum(pred_ngrams.values())
        scores.append(clipped / max(total, 1))

    if not scores or all(s == 0 for s in scores):
        return 0.0

    # Geometric mean of n-gram precisions
    log_avg = sum(math.log(s) for s in scores if s > 0) / len(scores)
    return round(bp * math.exp(log_avg), 4)


# ---------------------------------------------------------------------------
# RAGEvaluator class
# ---------------------------------------------------------------------------

class RAGEvaluator:
    """
    Evaluates a single RAG system output and compares multiple systems.

    Example
    -------
    >>> evaluator = RAGEvaluator(embedder=my_embedder)
    >>> result = evaluator.evaluate(
    ...     query="What is diabetes?",
    ...     answer="Diabetes is a metabolic disease...",
    ...     context="Diabetes mellitus is a group of diseases...",
    ...     ground_truth="Diabetes is a disease affecting blood sugar.",
    ... )
    >>> print(result)
    """

    def __init__(self, embedder=None):
        """
        Args:
            embedder: Optional Embedder instance for dense relevance scoring.
        """
        self.embedder = embedder

    def evaluate(
        self,
        query: str,
        answer: str,
        context: str,
        ground_truth: Optional[str] = None,
    ) -> dict:
        """
        Compute all metrics for a single (query, answer, context) triple.

        Args:
            query: User question.
            answer: System-generated answer.
            context: Retrieved context documents concatenated.
            ground_truth: Optional reference answer for F1/EM/BLEU.

        Returns:
            Dict with keys: faithfulness, relevance, f1, exact_match, bleu.
        """
        metrics = {
            "faithfulness": faithfulness_score(answer, context),
            "relevance": relevance_score(query, answer, self.embedder),
        }
        if ground_truth:
            metrics["f1"] = f1_score(answer, ground_truth)
            metrics["exact_match"] = exact_match(answer, ground_truth)
            metrics["bleu"] = bleu_score(answer, ground_truth)
        else:
            metrics["f1"] = None
            metrics["exact_match"] = None
            metrics["bleu"] = None
        return metrics

    def compare_systems(self, results: list) -> pd.DataFrame:
        """
        Build a comparison DataFrame from a list of per-system result dicts.

        Each dict in `results` should have keys:
            system_name, faithfulness, relevance, f1, latency

        Args:
            results: List of result dicts (one per architecture).

        Returns:
            Pandas DataFrame sorted by F1 descending.
        """
        if not results:
            return pd.DataFrame()

        rows = []
        for r in results:
            rows.append({
                "Architecture": r.get("system_name", "Unknown"),
                "F1": r.get("f1", 0.0) or 0.0,
                "Faithfulness": r.get("faithfulness", 0.0) or 0.0,
                "Relevance": r.get("relevance", 0.0) or 0.0,
                "Latency (s)": r.get("latency", 0.0) or 0.0,
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("F1", ascending=False).reset_index(drop=True)
        return df
