# evaluation/__init__.py
# Marks the evaluation directory as a Python package.
# All public evaluation utilities are importable from their sub-modules:
#   - evaluation.metrics   : RAGEvaluator and standalone metric functions
#   - evaluation.benchmark : AgenticRAGBenchmark

from evaluation.metrics import (
    faithfulness_score,
    relevance_score,
    f1_score,
    exact_match,
    bleu_score,
    RAGEvaluator,
)

__all__ = [
    "faithfulness_score",
    "relevance_score",
    "f1_score",
    "exact_match",
    "bleu_score",
    "RAGEvaluator",
]
