#!/usr/bin/env python3
"""
run_all.py — Demo all Agentic RAG architectures from arXiv:2501.09136

Usage:
    python run_all.py
    python run_all.py --query "What are treatments for Type 2 diabetes?"
    python run_all.py --model google/flan-t5-large
    python run_all.py --benchmark
    python run_all.py --arch naive_rag
"""

import argparse
import logging
import sys
import os
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress noisy warnings
logging.basicConfig(level=logging.WARNING)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── Rich terminal output ──────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import print as rprint
    RICH = True
    console = Console()
except ImportError:
    RICH = False
    console = None

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
MAGENTA= "\033[95m"
DIM    = "\033[2m"

def _banner():
    lines = [
        "╔══════════════════════════════════════════════════════════════════╗",
        "║    🤖  AGENTIC RAG IMPLEMENTATIONS  ·  arXiv:2501.09136          ║",
        "║    Survey: Agentic Retrieval-Augmented Generation                ║",
        "║    Singh et al. 2025                                             ║",
        "╚══════════════════════════════════════════════════════════════════╝",
    ]
    print(CYAN + "\n".join(lines) + RESET)


def _section(title: str, emoji: str = "▶"):
    print(f"\n{BOLD}{YELLOW}{'─'*65}{RESET}")
    print(f"{BOLD}{YELLOW}{emoji}  {title}{RESET}")
    print(f"{BOLD}{YELLOW}{'─'*65}{RESET}")


def _result(name: str, answer: str, meta: dict, latency: float):
    short_answer = answer.strip()[:180] + ("…" if len(answer) > 180 else "")
    print(f"{GREEN}  ✅ {name}{RESET}")
    print(f"{DIM}     Answer : {short_answer}{RESET}")
    if meta:
        meta_str = " | ".join(f"{k}: {v}" for k, v in list(meta.items())[:3])
        print(f"{DIM}     Info   : {meta_str}{RESET}")
    print(f"{DIM}     Latency: {latency:.2f}s{RESET}")


def _safe_run(name, fn, query):
    """Run an architecture, catching and reporting errors."""
    try:
        t0 = time.perf_counter()
        result = fn(query)
        elapsed = time.perf_counter() - t0
        answer = (
            result.get("answer")
            or result.get("final_answer")
            or result.get("synthesized_answer")
            or result.get("output")
            or str(result)
        )
        meta = {k: v for k, v in result.items()
                if k not in ("answer","final_answer","synthesized_answer",
                             "retrieved_docs","final_docs","source_chunks",
                             "history","sub_results","steps_output") and v}
        _result(name, str(answer), meta, elapsed)
        return result
    except Exception as e:
        print(f"  ❌ {name} — ERROR: {e}")
        return None


# ── Infrastructure setup ──────────────────────────────────────────────────────

def setup_infrastructure(model_name: str):
    from core.embeddings import Embedder
    from core.vector_store import FAISSVectorStore
    from core.llm import LocalLLM
    from core.graph_store import KnowledgeGraph
    from data.sample_documents import DOCUMENTS

    print(f"\n{BOLD}🔧 Loading infrastructure...{RESET}")
    print(f"   LLM model  : {model_name}")
    print(f"   Embedder   : all-MiniLM-L6-v2")
    print(f"   Documents  : {len(DOCUMENTS)} synthetic docs across 5 domains")

    embedder = Embedder()
    llm = LocalLLM(model_name=model_name)
    vector_store = FAISSVectorStore(embedder)
    vector_store.add_documents(DOCUMENTS)

    kg = KnowledgeGraph()
    kg.build_from_documents([d["text"] for d in DOCUMENTS])

    print(f"   ✅ Indexed {len(DOCUMENTS)} documents, KG: {len(kg.graph.nodes())} entities")
    return embedder, llm, vector_store, kg


# ── Architecture runners ──────────────────────────────────────────────────────

def run_rag_paradigms(llm, vector_store, kg, query):
    _section("RAG PARADIGMS (§2.3)", "📚")

    from rag_paradigms.naive_rag import NaiveRAG
    from rag_paradigms.advanced_rag import AdvancedRAG
    from rag_paradigms.modular_rag import ModularRAG
    from rag_paradigms.graph_rag import GraphRAG

    _safe_run("Naïve RAG (§2.3.1)",   NaiveRAG(vector_store, llm, k=3).query, query)
    _safe_run("Advanced RAG (§2.3.2)", AdvancedRAG(vector_store, llm, k=3).query, query)
    _safe_run("Modular RAG (§2.3.3)",  ModularRAG(vector_store, llm).query, query)
    _safe_run("Graph RAG (§2.3.4)",    GraphRAG(vector_store, llm, kg).query, query)


def run_agentic_patterns(llm, vector_store, query):
    _section("AGENTIC DESIGN PATTERNS (§3)", "🧠")

    from agentic_patterns.reflection import ReflectionAgent
    from agentic_patterns.planning import PlanningAgent
    from agentic_patterns.tool_use import ToolUseAgent
    from agentic_patterns.multi_agent import MultiAgentOrchestrator

    _safe_run("Reflection (§3.1)",   ReflectionAgent(llm, vector_store, max_iterations=2).query, query)
    _safe_run("Planning (§3.2)",     PlanningAgent(llm, vector_store).query, query)
    _safe_run("Tool Use (§3.3)",     ToolUseAgent(llm, vector_store).query, query)
    _safe_run("Multi-Agent (§3.4)",  MultiAgentOrchestrator(llm, vector_store).query, query)


def run_workflow_patterns(llm, vector_store, query):
    _section("WORKFLOW PATTERNS (§4)", "⚙️")

    from workflow_patterns.prompt_chaining import PromptChaining
    from workflow_patterns.routing import RoutingWorkflow
    from workflow_patterns.parallelization import ParallelWorkflow
    from workflow_patterns.orchestrator_workers import OrchestratorAgent
    from workflow_patterns.evaluator_optimizer import EvaluatorOptimizer

    _safe_run("Prompt Chaining (§4.1)",       PromptChaining(llm, vector_store).run, query)
    _safe_run("Routing (§4.2)",               RoutingWorkflow(llm, vector_store).query, query)
    _safe_run("Parallelization (§4.3)",       ParallelWorkflow(llm, vector_store).run_parallel, query)
    _safe_run("Orchestrator-Workers (§4.4)",  OrchestratorAgent(llm, vector_store).query, query)
    _safe_run("Evaluator-Optimizer (§4.5)",   EvaluatorOptimizer(llm, vector_store).query, query)


def run_taxonomy(llm, vector_store, kg, embedder, query):
    _section("TAXONOMY OF AGENTIC RAG SYSTEMS (§5)", "🏛️")

    from taxonomy.single_agent_rag import SingleAgentRAG
    from taxonomy.multi_agent_rag import MultiAgentRAG
    from taxonomy.hierarchical_rag import HierarchicalRAG
    from taxonomy.corrective_rag import CorrectiveRAG
    from taxonomy.adaptive_rag import AdaptiveRAG
    from taxonomy.agent_g import AgentG
    from taxonomy.gear import GeAR
    from taxonomy.document_workflow_rag import DocumentWorkflowRAG
    from data.sample_documents import DOCUMENTS

    _safe_run("Single-Agent Router (§5.1)", SingleAgentRAG(llm, vector_store).query, query)
    _safe_run("Multi-Agent RAG (§5.2)",     MultiAgentRAG(llm, vector_store).query, query)
    _safe_run("Hierarchical RAG (§5.3)",    HierarchicalRAG(llm, vector_store).query, query)
    _safe_run("Corrective RAG (§5.4)",      CorrectiveRAG(llm, vector_store).query, query)
    _safe_run("Adaptive RAG (§5.5)",        AdaptiveRAG(llm, vector_store).query, query)
    _safe_run("Agent-G (§5.6.1)",           AgentG(llm, vector_store, kg).query, query)
    _safe_run("GeAR (§5.6.2)",              GeAR(llm, vector_store, kg).query, query)

    # Document workflow needs ingestion first
    doc_rag = DocumentWorkflowRAG(llm, embedder)
    doc_rag.process_documents(DOCUMENTS)
    _safe_run("Agentic Doc Workflow (§5.7)", doc_rag.query, query)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run all Agentic RAG architectures from arXiv:2501.09136"
    )
    parser.add_argument(
        "--query", "-q",
        default="What are the key treatments for Type 2 diabetes and their mechanisms?",
        help="Question to answer with all architectures."
    )
    parser.add_argument(
        "--model", "-m",
        default="google/flan-t5-base",
        help="HuggingFace model name (default: google/flan-t5-base)."
    )
    parser.add_argument(
        "--benchmark", "-b",
        action="store_true",
        help="Run full benchmark comparison (slower)."
    )
    parser.add_argument(
        "--arch", "-a",
        choices=["rag_paradigms", "patterns", "workflows", "taxonomy", "all"],
        default="all",
        help="Which group of architectures to run."
    )
    args = parser.parse_args()

    _banner()
    print(f"\n{BOLD}Query:{RESET} {args.query}\n")

    if args.benchmark:
        print(f"{YELLOW}Running full benchmark...{RESET}")
        from evaluation.benchmark import AgenticRAGBenchmark
        bench = AgenticRAGBenchmark(model_name=args.model)
        results = bench.run_all(max_queries=3)
        report = bench.generate_report(results)
        print(report)
        return

    embedder, llm, vector_store, kg = setup_infrastructure(args.model)

    if args.arch in ("rag_paradigms", "all"):
        run_rag_paradigms(llm, vector_store, kg, args.query)

    if args.arch in ("patterns", "all"):
        run_agentic_patterns(llm, vector_store, args.query)

    if args.arch in ("workflows", "all"):
        run_workflow_patterns(llm, vector_store, args.query)

    if args.arch in ("taxonomy", "all"):
        run_taxonomy(llm, vector_store, kg, embedder, args.query)

    print(f"\n{GREEN}{BOLD}✅ All architectures completed!{RESET}")
    print(f"{DIM}Set this as your workspace: {os.path.abspath('.')}{RESET}\n")


if __name__ == "__main__":
    main()
