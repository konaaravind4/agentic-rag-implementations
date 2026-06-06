"""
evaluation/benchmark.py
========================
Comprehensive benchmark runner for all Agentic RAG architectures.

'Agentic RAG: A Survey' (arXiv:2501.09136)

``AgenticRAGBenchmark`` initialises every implemented architecture, runs
each one on a shared set of queries, collects metrics, and produces a
formatted comparison report.

Usage
-----
    # Programmatic:
    from evaluation.benchmark import AgenticRAGBenchmark
    bench = AgenticRAGBenchmark(llm, vector_store, knowledge_graph, embedder)
    results = bench.run_all(queries)
    print(bench.generate_report(results))

    # CLI:
    python evaluation/benchmark.py
"""

import time
import logging
import sys
import os
import importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import RAGEvaluator, faithfulness_score, f1_score, bleu_score

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Optional pandas
# ---------------------------------------------------------------------------
try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False


def _try_import(module_path, class_name):
    """Attempt to import a class; return None on failure."""
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    except Exception as exc:
        logger.debug("Could not import %s.%s: %s", module_path, class_name, exc)
        return None


def _build_context_string(retrieved_docs: list) -> str:
    """Concatenate retrieved doc texts into a single context string."""
    if not retrieved_docs:
        return ""
    parts = []
    for d in retrieved_docs:
        if isinstance(d, dict):
            parts.append(d.get("text", ""))
        elif isinstance(d, str):
            parts.append(d)
    return " ".join(p for p in parts if p)


class AgenticRAGBenchmark:
    """
    Runs all available RAG architectures on a shared query set and produces
    a formatted comparison report.

    Architecture registry (attempts to load; skips gracefully on import error):
      - taxonomy.single_agent_rag   : SingleAgentRAG   (§5.1)
      - taxonomy.adaptive_rag       : AdaptiveRAG       (§5.5)
      - taxonomy.agent_g            : AgentG            (§5.6.1)
      - taxonomy.gear               : GeAR              (§5.6.2)
      - taxonomy.document_workflow_rag : DocumentWorkflowRAG (§5.7)
      - rag_paradigms.naive_rag     : NaiveRAG
      - agentic_patterns.reflection : ReflectionAgent

    Parameters
    ----------
    llm : LocalLLM
        Shared language model for all architectures.
    vector_store : FAISSVectorStore
        Pre-indexed vector store.
    knowledge_graph : KnowledgeGraph
        Populated knowledge graph.
    embedder : Embedder
        Shared text embedder.
    ground_truths : list[str], optional
        Reference answers aligned with the query list.
    """

    # (module_path, class_name, section_label)
    _REGISTRY = [
        ("taxonomy.single_agent_rag",      "SingleAgentRAG",      "§5.1"),
        ("taxonomy.adaptive_rag",          "AdaptiveRAG",         "§5.5"),
        ("taxonomy.agent_g",               "AgentG",              "§5.6.1"),
        ("taxonomy.gear",                  "GeAR",                "§5.6.2"),
        ("taxonomy.document_workflow_rag", "DocumentWorkflowRAG", "§5.7"),
        ("rag_paradigms.naive_rag",        "NaiveRAG",            "§2.3.1"),
        ("agentic_patterns.reflection",    "ReflectionAgent",     "§3.1"),
    ]

    def __init__(
        self,
        llm,
        vector_store,
        knowledge_graph,
        embedder,
        ground_truths=None,
    ):
        self.llm             = llm
        self.vector_store    = vector_store
        self.knowledge_graph = knowledge_graph
        self.embedder        = embedder
        self.ground_truths   = ground_truths or []
        self.evaluator       = RAGEvaluator(embedder=embedder)
        self._systems        = {}
        self._build_systems()

    # ------------------------------------------------------------------
    # System construction
    # ------------------------------------------------------------------

    def _build_systems(self):
        """Attempt to instantiate every registered architecture."""
        for module_path, class_name, section in self._REGISTRY:
            cls = _try_import(module_path, class_name)
            if cls is None:
                continue
            try:
                instance = self._instantiate(cls, class_name)
                if instance is not None:
                    label = f"{class_name} ({section})"
                    self._systems[label] = instance
                    logger.info("[Benchmark] Loaded: %s", label)
            except Exception as exc:
                logger.warning("[Benchmark] Could not create %s: %s", class_name, exc)

    def _instantiate(self, cls, class_name):
        """
        Try different argument combinations to construct the class.
        Returns None if all attempts fail.
        """
        # Different architectures take different constructor args
        arg_sets = [
            dict(llm=self.llm, vector_store=self.vector_store,
                 knowledge_graph=self.knowledge_graph),
            dict(llm=self.llm, vector_store=self.vector_store,
                 use_llm_classifier=False),
            dict(llm=self.llm, vector_store=self.vector_store),
            dict(vector_store=self.vector_store, llm=self.llm),
        ]
        for kwargs in arg_sets:
            try:
                return cls(**kwargs)
            except TypeError:
                continue
        return None

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    def run_all(self, queries=None, max_queries=None):
        """
        Run all architectures on the given queries.

        Parameters
        ----------
        queries : list[str], optional
            Queries to evaluate. Falls back to SAMPLE_QUERIES.
        max_queries : int, optional
            Cap the number of queries.

        Returns
        -------
        dict
            {
              architecture_name: {
                query_results, avg_f1, avg_faithfulness,
                avg_relevance, avg_bleu, avg_latency,
                n_queries, n_errors
              }, ...
            }
        """
        if queries is None:
            from data.sample_documents import SAMPLE_QUERIES
            queries = SAMPLE_QUERIES

        if max_queries:
            queries = queries[:max_queries]

        gt_list = (
            self.ground_truths[:len(queries)]
            if self.ground_truths
            else [""] * len(queries)
        )

        all_results = {}
        for name, system in self._systems.items():
            print(f"\n  ▶ Running {name} on {len(queries)} queries…")
            arch_res = self._run_single(name, system, queries, gt_list)
            all_results[name] = arch_res

        return all_results

    def _run_single(self, name, system, queries, ground_truths):
        """Run one architecture on all queries."""
        query_results = []
        n_errors = 0

        for idx, (q, gt) in enumerate(zip(queries, ground_truths)):
            try:
                t0 = time.perf_counter()
                raw = system.query(q)
                latency = time.perf_counter() - t0

                answer = (
                    raw.get("answer")
                    or raw.get("final_answer")
                    or raw.get("response")
                    or str(raw)
                )
                context = _build_context_string(
                    raw.get("retrieved_docs",
                    raw.get("source_chunks",
                    raw.get("expanded_results", [])))
                )

                faith = faithfulness_score(str(answer), context) if context else 0.0
                f1    = f1_score(str(answer), gt) if gt else 0.0
                bl    = bleu_score(str(answer), gt) if gt else 0.0

                query_results.append({
                    "query_idx":    idx,
                    "query":        q,
                    "answer":       str(answer)[:300],
                    "latency":      round(latency, 4),
                    "faithfulness": round(faith, 4),
                    "f1":           round(f1, 4),
                    "bleu":         round(bl, 4),
                })
                print(f"    ✓ [{latency:.1f}s] {q[:55]}")

            except Exception as exc:
                n_errors += 1
                logger.warning("[%s] Q%d failed: %s", name, idx, exc)
                query_results.append({
                    "query_idx":    idx,
                    "query":        q,
                    "answer":       f"ERROR: {exc}",
                    "latency":      0.0,
                    "faithfulness": 0.0,
                    "f1":           0.0,
                    "bleu":         0.0,
                    "error":        str(exc),
                })

        n = len(query_results)
        def _mean(key):
            vals = [r[key] for r in query_results
                    if isinstance(r.get(key), (int, float))]
            return round(sum(vals) / max(len(vals), 1), 4)

        return {
            "query_results":    query_results,
            "avg_f1":           _mean("f1"),
            "avg_faithfulness": _mean("faithfulness"),
            "avg_bleu":         _mean("bleu"),
            "avg_latency":      _mean("latency"),
            "n_queries":        n,
            "n_errors":         n_errors,
        }

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self, results: dict) -> str:
        """
        Produce a formatted ASCII comparison table.

        Columns: Architecture | Avg F1 | Faithfulness | Relevance | Avg Latency(s)

        Parameters
        ----------
        results : dict
            Output of run_all().

        Returns
        -------
        str
            Multi-line formatted report.
        """
        if not results:
            return "No benchmark results available."

        rows = []
        for name, stats in results.items():
            rows.append({
                "Architecture":    name,
                "Avg F1":          stats.get("avg_f1", 0.0),
                "Faithfulness":    stats.get("avg_faithfulness", 0.0),
                "Avg BLEU":        stats.get("avg_bleu", 0.0),
                "Avg Latency(s)":  stats.get("avg_latency", 0.0),
                "Errors":          stats.get("n_errors", 0),
            })
        rows.sort(key=lambda x: x["Avg F1"], reverse=True)

        # Column widths
        col_specs = [
            ("Architecture",   28),
            ("Avg F1",          8),
            ("Faithfulness",   14),
            ("Avg BLEU",        9),
            ("Avg Latency(s)", 14),
            ("Errors",          7),
        ]
        sep = "  ".join("─" * w for _, w in col_specs)
        header_row = "  ".join(
            h.ljust(w) if h != "Architecture" else h.ljust(w)
            for h, w in col_specs
        )

        lines = [
            "",
            "═" * 72,
            "   AGENTIC RAG BENCHMARK REPORT  |  arXiv:2501.09136",
            "═" * 72,
            "",
            header_row,
            sep,
        ]

        for r in rows:
            cells = [
                str(r["Architecture"])[:28].ljust(28),
                f"{r['Avg F1']:.4f}".ljust(8),
                f"{r['Faithfulness']:.4f}".ljust(14),
                f"{r['Avg BLEU']:.4f}".ljust(9),
                f"{r['Avg Latency(s)']:.3f}".ljust(14),
                str(r["Errors"]).ljust(7),
            ]
            lines.append("  ".join(cells))

        lines += [sep, "═" * 72, ""]

        if rows:
            best = rows[0]
            lines.append(
                f"  🏆  Best by Avg F1: {best['Architecture']} "
                f"(F1={best['Avg F1']:.4f})"
            )
            lines.append("")

        return "\n".join(lines)

    def to_dataframe(self, results):
        """Return results as a sorted pandas DataFrame (requires pandas)."""
        data = []
        for arch, stats in results.items():
            data.append({
                "Architecture": arch,
                "Avg F1":       stats.get("avg_f1", 0.0),
                "Faithfulness": stats.get("avg_faithfulness", 0.0),
                "Avg BLEU":     stats.get("avg_bleu", 0.0),
                "Avg Latency":  stats.get("avg_latency", 0.0),
                "N Queries":    stats.get("n_queries", 0),
                "N Errors":     stats.get("n_errors", 0),
            })
        if _PANDAS:
            import pandas as pd
            return pd.DataFrame(data).sort_values("Avg F1", ascending=False).reset_index(drop=True)
        return sorted(data, key=lambda r: r["Avg F1"], reverse=True)


# ---------------------------------------------------------------------------
# Standalone CLI entry point
# ---------------------------------------------------------------------------

def _build_mock_infrastructure():
    """Lightweight mock infrastructure (no model download needed)."""
    import numpy as np

    class _MockEmbedder:
        embedding_dim = 8
        def embed(self, texts, **kw):
            return np.random.rand(len(texts), 8).astype("float32")
        def embed_query(self, q, **kw):
            return np.random.rand(8).astype("float32")

    class _MockLLM:
        model_name = "mock"
        def generate(self, prompt, **kw):
            if "what is" in prompt.lower():
                return "It is a core concept in AI and information retrieval."
            if "how" in prompt.lower():
                return "It works by combining retrieval with generation."
            return "This is a synthesised answer from the retrieved context."
        def score_relevance(self, q, d):
            return 0.75

    class _MockVectorStore:
        def __init__(self):
            self._docs = [
                {"text": "RAG combines retrieval with LLM generation.", "score": 0.9,
                 "metadata": {"doc_id": "d1", "chunk_id": "chunk_0"},
                 "doc_id": "d1", "chunk_id": "chunk_0"},
                {"text": "LLMs are transformer-based language models.", "score": 0.85,
                 "metadata": {"doc_id": "d2", "chunk_id": "chunk_0"},
                 "doc_id": "d2", "chunk_id": "chunk_0"},
                {"text": "Graph RAG uses knowledge graphs for retrieval.", "score": 0.8,
                 "metadata": {"doc_id": "d3", "chunk_id": "chunk_0"},
                 "doc_id": "d3", "chunk_id": "chunk_0"},
            ]
        def search(self, query, top_k=5):
            return self._docs[:top_k]
        def similarity_search(self, query, k=5):
            return self._docs[:k]
        def add_documents(self, texts, metadatas=None, doc_ids=None):
            pass
        def add_texts(self, texts, metadatas=None):
            pass
        @property
        def num_documents(self):
            return len(self._docs)

    from core.graph_store import build_demo_knowledge_graph
    kg = build_demo_knowledge_graph()
    return _MockLLM(), _MockVectorStore(), kg, _MockEmbedder()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    print("\n🚀 Agentic RAG Benchmark — arXiv:2501.09136")
    print("   Using mock LLM & vector store (no model download)\n")

    from data.sample_documents import SAMPLE_QUERIES, QUERY_GROUND_TRUTHS

    llm, vs, kg, embedder = _build_mock_infrastructure()
    demo_queries = SAMPLE_QUERIES[:5]
    demo_gt      = [QUERY_GROUND_TRUTHS.get(i, "") for i in range(len(demo_queries))]

    bench = AgenticRAGBenchmark(
        llm=llm,
        vector_store=vs,
        knowledge_graph=kg,
        embedder=embedder,
        ground_truths=demo_gt,
    )

    print(f"Architectures loaded: {list(bench._systems.keys())}\n")
    results = bench.run_all(demo_queries)
    report  = bench.generate_report(results)
    print(report)

    if _PANDAS:
        df = bench.to_dataframe(results)
        print("\nPandas DataFrame:")
        print(df.to_string(index=False))
