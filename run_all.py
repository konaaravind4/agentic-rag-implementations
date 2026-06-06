#!/usr/bin/env python3
"""
run_all.py
==========
Complete demo script for all Agentic RAG architectures.

'Agentic RAG: A Survey' (arXiv:2501.09136)

Usage
-----
    python run_all.py                               # Default demo query
    python run_all.py --query "What is Graph RAG?" # Custom query
    python run_all.py --model google/flan-t5-base  # Specific HuggingFace model
    python run_all.py --all                         # Run all 20 benchmark queries
    python run_all.py --mock                        # Use mock LLM (no download)
    python run_all.py --query "..." --mock          # Custom query with mock LLM

Each architecture is run on the demo query and its result is printed in a
formatted block showing: architecture name, answer excerpt, key metadata,
and latency.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)


# ===========================================================================
# ANSI colour helpers
# ===========================================================================

class _C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    MAGENTA = "\033[95m"
    RED     = "\033[91m"
    BLUE    = "\033[94m"
    GREY    = "\033[90m"

    @staticmethod
    def code(text: str, *codes: str) -> str:
        return "".join(codes) + text + _C.RESET


def _bold(t):     return _C.code(t, _C.BOLD)
def _cyan(t):     return _C.code(t, _C.CYAN)
def _green(t):    return _C.code(t, _C.GREEN)
def _yellow(t):   return _C.code(t, _C.YELLOW)
def _magenta(t):  return _C.code(t, _C.MAGENTA)
def _blue(t):     return _C.code(t, _C.BLUE)
def _grey(t):     return _C.code(t, _C.GREY)
def _red(t):      return _C.code(t, _C.RED)


# ===========================================================================
# Mock infrastructure (when --mock or no model available)
# ===========================================================================

class _MockEmbedder:
    """Deterministic hash-based pseudo-embedder (no model download)."""
    embedding_dim = 16

    def embed(self, texts, **kw):
        import hashlib
        import numpy as np
        rows = []
        for text in texts:
            raw = hashlib.sha256(text.encode()).digest()
            arr = (list(raw) * (self.embedding_dim // len(raw) + 1))[:self.embedding_dim]
            v = np.array([b / 255.0 - 0.5 for b in arr], dtype="float32")
            norm = np.linalg.norm(v)
            rows.append(v / norm if norm > 0 else v)
        return np.stack(rows)

    def embed_query(self, query, **kw):
        return self.embed([query])[0]


class _MockLLM:
    """Lightweight template-based mock LLM with no model download."""
    model_name = "mock-llm"

    _TEMPLATES = {
        "what is":  "It is a foundational concept — {q} refers to a key method "
                    "in modern AI and information retrieval systems.",
        "how does": "The process works through a multi-step pipeline: first "
                    "relevant information is retrieved, then synthesised by the model.",
        "compare":  "The two approaches differ in: (1) retrieval strategy — one uses "
                    "dense vectors while the other uses graphs, (2) latency, "
                    "(3) coverage.",
        "why":      "The reason is multi-faceted. First, grounded generation; "
                    "second, hallucination reduction through evidence-based retrieval.",
        "default":  "Based on the retrieved context: this relates to {q}, "
                    "which involves retrieval-augmented approaches combining "
                    "neural and symbolic methods.",
    }

    def generate(self, prompt: str, **kw) -> str:
        pl = prompt.lower()
        q = ""
        for line in prompt.split("\n"):
            if "question:" in line.lower():
                q = line.split(":", 1)[-1].strip()[:60]
                break
        for key, tmpl in self._TEMPLATES.items():
            if key in pl:
                return tmpl.format(q=q or "the query")
        return self._TEMPLATES["default"].format(q=q or "the query")

    def score_relevance(self, query: str, document: str) -> float:
        q_toks = set(query.lower().split())
        d_toks = set(document.lower().split())
        if not q_toks:
            return 0.5
        return min(1.0, len(q_toks & d_toks) / len(q_toks))


class _MockVectorStore:
    """In-memory store with pre-loaded sample documents."""

    def __init__(self):
        from data.sample_documents import SAMPLE_DOCUMENTS
        self._docs = [
            {
                "id":       doc["id"],
                "text":     doc["text"],
                "score":    0.85,
                "metadata": {**doc.get("metadata", {}),
                             "doc_id": doc["id"],
                             "chunk_id": "chunk_0"},
                "doc_id":   doc["id"],
                "chunk_id": "chunk_0",
            }
            for doc in SAMPLE_DOCUMENTS
        ]

    def _rank(self, query: str) -> list:
        q_words = set(query.lower().split())
        def score(d):
            d_words = set(d["text"].lower().split())
            return len(q_words & d_words)
        return sorted(self._docs, key=score, reverse=True)

    def search(self, query, top_k=5):
        return self._rank(query)[:top_k]

    def similarity_search(self, query, k=5):
        return self._rank(query)[:k]

    def add_documents(self, texts, metadatas=None, doc_ids=None):
        pass

    def add_texts(self, texts, metadatas=None):
        pass

    @property
    def num_documents(self):
        return len(self._docs)


# ===========================================================================
# Infrastructure builder
# ===========================================================================

def build_infrastructure(model_name: str, use_mock: bool):
    """
    Construct shared Embedder, LLM, VectorStore, and KnowledgeGraph.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (when use_mock=False).
    use_mock : bool
        If True, use lightweight mock instances.

    Returns
    -------
    tuple : (llm, vector_store, knowledge_graph, embedder)
    """
    from core.graph_store import build_demo_knowledge_graph
    knowledge_graph = build_demo_knowledge_graph()

    if use_mock:
        print(f"  {_yellow('⚡')} Using mock LLM + embedder (no model download)")
        llm          = _MockLLM()
        embedder     = _MockEmbedder()
        vector_store = _MockVectorStore()
    else:
        print(f"  {_cyan('🤖')} Loading embedder…")
        from core.embeddings import Embedder
        embedder = Embedder()

        print(f"  {_cyan('🤖')} Loading LLM ({model_name})…")
        from core.llm import LocalLLM
        llm = LocalLLM(model_name=model_name)

        print(f"  {_cyan('📚')} Building vector store & indexing documents…")
        from core.vector_store import FAISSVectorStore
        from data.sample_documents import SAMPLE_DOCUMENTS
        vector_store = FAISSVectorStore(embedder)
        try:
            vector_store.add_documents(SAMPLE_DOCUMENTS)
        except Exception:
            texts = [d["text"] for d in SAMPLE_DOCUMENTS]
            metas = [d.get("metadata", {}) for d in SAMPLE_DOCUMENTS]
            vector_store.add_documents(texts, metadatas=metas)

    print(f"  {_green('✅')} Infrastructure ready. KG: {knowledge_graph}\n")
    return llm, vector_store, knowledge_graph, embedder


# ===========================================================================
# Architecture registry
# ===========================================================================

ARCHITECTURE_REGISTRY = [
    # (display_name, emoji, module_path, class_name, section, kwargs_fn)
    (
        "Single-Agent RAG (Router)", "🤖",
        "taxonomy.single_agent_rag", "SingleAgentRAG", "§5.1",
        lambda llm, vs, kg, emb: dict(llm=llm, vector_store=vs),
    ),
    (
        "Adaptive RAG", "🔀",
        "taxonomy.adaptive_rag", "AdaptiveRAG", "§5.5",
        lambda llm, vs, kg, emb: dict(llm=llm, vector_store=vs, use_llm_classifier=False),
    ),
    (
        "Agent-G (Graph + Text)", "🌐",
        "taxonomy.agent_g", "AgentG", "§5.6.1",
        lambda llm, vs, kg, emb: dict(llm=llm, vector_store=vs, knowledge_graph=kg),
    ),
    (
        "GeAR (BM25 + Graph Expansion)", "📈",
        "taxonomy.gear", "GeAR", "§5.6.2",
        lambda llm, vs, kg, emb: dict(llm=llm, knowledge_graph=kg),
    ),
    (
        "Document Workflow RAG", "📄",
        "taxonomy.document_workflow_rag", "DocumentWorkflowRAG", "§5.7",
        lambda llm, vs, kg, emb: dict(llm=llm, vector_store=vs),
    ),
    (
        "Naive RAG", "📋",
        "rag_paradigms.naive_rag", "NaiveRAG", "§2.3.1",
        lambda llm, vs, kg, emb: dict(vector_store=vs, llm=llm),
    ),
    (
        "Reflection Agent", "🔄",
        "agentic_patterns.reflection", "ReflectionAgent", "§3.1",
        lambda llm, vs, kg, emb: dict(llm=llm, vector_store=vs),
    ),
]


def _try_import(module_path: str, class_name: str):
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    except Exception as exc:
        logger.debug("Import failed %s.%s: %s", module_path, class_name, exc)
        return None


def _run_one_architecture(
    display_name, emoji, module_path, class_name, section,
    args_fn, llm, vs, kg, embedder, query,
) -> Optional[Dict[str, Any]]:
    """Instantiate one architecture and run a single query."""
    cls = _try_import(module_path, class_name)
    if cls is None:
        return None
    try:
        kwargs   = args_fn(llm, vs, kg, embedder)
        instance = cls(**kwargs)

        # Architecture-specific setup
        if class_name == "DocumentWorkflowRAG":
            from data.sample_documents import SAMPLE_DOCUMENTS
            instance.process_documents(SAMPLE_DOCUMENTS[:10])

        if class_name == "GeAR":
            from data.sample_documents import SAMPLE_DOCUMENTS
            corpus = [{"id": d["id"], "text": d["text"]} for d in SAMPLE_DOCUMENTS]
            instance.index(corpus)

        t0     = time.perf_counter()
        result = instance.query(query)
        result.setdefault("latency", time.perf_counter() - t0)
        return result

    except Exception as exc:
        logger.warning("Architecture %s failed: %s", class_name, exc)
        return {"error": str(exc), "latency": 0.0}


# ===========================================================================
# Output formatting
# ===========================================================================

def _sep(char="─", w=72):
    print(_grey(char * w))


def _print_header(text: str):
    w = 72
    pad = max(0, w - len(text) - 4)
    print()
    print(_bold(_cyan("┌" + "─" * (w - 2) + "┐")))
    print(_bold(_cyan("│ ")) + _bold(text) + " " * pad + _bold(_cyan(" │")))
    print(_bold(_cyan("└" + "─" * (w - 2) + "┘")))
    print()


def _print_result(display_name, emoji, section, result, idx, total):
    _sep()
    num  = _grey(f"[{idx}/{total}]")
    arch = _bold(_magenta(f"{emoji}  {display_name}"))
    sect = _blue(section)
    print(f"  {num}  {arch}  {sect}")
    _sep("·")

    if "error" in result:
        print(f"  {_red('❌ Error:')} {result['error']}")
    else:
        answer = str(
            result.get("answer")
            or result.get("final_answer")
            or result.get("response")
            or "N/A"
        )
        excerpt = answer[:280] + ("…" if len(answer) > 280 else "")
        print(f"  {_bold('Answer:')}")
        words, line = excerpt.split(), "    "
        for word in words:
            if len(line) + len(word) + 1 > 72:
                print(line)
                line = "    " + word + " "
            else:
                line += word + " "
        if line.strip():
            print(line)

        meta = []
        for k, label in [
            ("complexity",          "complexity"),
            ("tool_used",           "tool"),
            ("retrieval_source",    "source"),
            ("feedback_iterations", "feedback"),
            ("expansion_steps",     "expansions"),
            ("rounds",              "rounds"),
            ("iterations",          "iterations"),
        ]:
            if k in result:
                meta.append(f"{label}={_yellow(str(result[k]))}")
        if "citations" in result:
            meta.append(f"citations={_yellow(str(result['citations'][:2]))}")
        n_docs = len(result.get("retrieved_docs",
                    result.get("source_chunks",
                    result.get("expanded_results", []))))
        if n_docs:
            meta.append(f"docs={_yellow(str(n_docs))}")
        if meta:
            print(f"\n  {_bold('Meta:')}  {_grey(' | ').join(meta)}")

        lat = result.get("latency", 0.0)
        print(f"  {_bold('Latency:')} {_green(f'{lat:.4f}s')}")
    print()


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Agentic RAG Demo — arXiv:2501.09136",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py
  python run_all.py --query "How does Adaptive RAG work?"
  python run_all.py --model google/flan-t5-base
  python run_all.py --all --mock
  python run_all.py --query "What is FAISS?" --mock
        """,
    )
    parser.add_argument(
        "--query", "-q",
        default=(
            "How does Retrieval-Augmented Generation reduce hallucination in LLMs?"
        ),
        help="The question to ask all architectures.",
    )
    parser.add_argument(
        "--model", "-m",
        default="google/flan-t5-base",
        help="HuggingFace model name (used when --mock is not set).",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all 20 benchmark queries (slow).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use a lightweight mock LLM/embedder (no model download).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    # Banner
    print()
    print(_bold(_cyan("╔" + "═" * 68 + "╗")))
    print(_bold(_cyan("║")) + _bold("   🤖  AGENTIC RAG IMPLEMENTATIONS DEMO                   ") + _bold(_cyan("║")))
    print(_bold(_cyan("║")) + _grey("   Paper: Agentic RAG: A Survey  |  arXiv:2501.09136      ") + _bold(_cyan("║")))
    print(_bold(_cyan("╚" + "═" * 68 + "╝")))
    print()

    # Queries
    if args.all:
        from data.sample_documents import SAMPLE_QUERIES
        queries = SAMPLE_QUERIES
        print(f"  {_yellow('📋')} Running ALL {len(queries)} benchmark queries.")
    else:
        queries = [args.query]

    print(f"  {_bold('Query:')} {_cyan(queries[0])}")
    if len(queries) > 1:
        print(f"         (+ {len(queries)-1} more)")
    print()

    # Infrastructure
    _print_header("🔧  Initialising Infrastructure")
    t0 = time.perf_counter()
    llm, vs, kg, embedder = build_infrastructure(args.model, args.mock)
    print(f"  Infrastructure ready in {_green(f'{time.perf_counter()-t0:.2f}s')}\n")

    # Run each architecture
    total = len(ARCHITECTURE_REGISTRY)
    all_summaries: List[Dict] = []

    for query in queries:
        if len(queries) > 1:
            _print_header(
                f"❓  Query: {query[:58]}{'…' if len(query)>58 else ''}"
            )

        for idx, (dname, emoji, mod, cls_name, sect, args_fn) in \
                enumerate(ARCHITECTURE_REGISTRY, 1):

            result = _run_one_architecture(
                dname, emoji, mod, cls_name, sect,
                args_fn, llm, vs, kg, embedder, query,
            )

            if result is None:
                print(f"  {_grey('⏭  Skipped')} {dname} {_grey('(import failed)')}")
                continue

            _print_result(dname, emoji, sect, result, idx, total)
            all_summaries.append({
                "architecture": dname,
                "query":        query,
                "latency":      result.get("latency", 0.0),
                "error":        "error" in result,
            })

    # Summary
    _sep("═")
    print(_bold("  📊  Run Summary"))
    _sep("─")
    arch_lats: Dict[str, List[float]] = {}
    for s in all_summaries:
        arch_lats.setdefault(s["architecture"], []).append(s["latency"])

    print(f"  {'Architecture':<35} {'Avg Latency':>12}  {'Status':>8}")
    print(f"  {'─'*35} {'─'*12}  {'─'*8}")
    for name, lats in arch_lats.items():
        avg_lat = sum(lats) / max(len(lats), 1)
        print(f"  {name:<35} {avg_lat:>10.4f}s  {_green('✓ OK')}")

    print()
    _sep("═")
    print()
    print(
        f"  {_green('✅')} Demo complete. "
        f"Ran {len(all_summaries)} architecture×query combinations.\n"
    )


if __name__ == "__main__":
    main()
