# Agentic RAG Implementations

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2501.09136-red)](https://arxiv.org/abs/2501.09136)

> **Complete, runnable implementation of every architecture from the survey paper:**  
> *"Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG"* — Singh et al. (2025)  
> arXiv: [2501.09136](https://arxiv.org/abs/2501.09136)

---

##  What's Implemented

This repository implements **all models and architectures** described in the paper:

### RAG Paradigms (§2.3)
| Module | Architecture | Description |
|--------|-------------|-------------|
| `rag_paradigms/naive_rag.py` | **Naïve RAG** | Simple retrieve-then-generate |
| `rag_paradigms/advanced_rag.py` | **Advanced RAG** | HyDE, query rewriting, reranking, context compression |
| `rag_paradigms/modular_rag.py` | **Modular RAG** | Plug-and-play pipeline modules |
| `rag_paradigms/graph_rag.py` | **Graph RAG** | Knowledge graph-augmented retrieval |
| `rag_paradigms/agentic_rag_base.py` | **Agentic RAG Base** | Abstract base with agentic utilities |

### Agentic Design Patterns (§3)
| Module | Pattern | Description |
|--------|---------|-------------|
| `agentic_patterns/reflection.py` | **Reflection** | Self-critique & iterative refinement (Self-Refine style) |
| `agentic_patterns/planning.py` | **Planning** | Task decomposition & sub-query solving |
| `agentic_patterns/tool_use.py` | **Tool Use** | Dynamic tool selection & execution |
| `agentic_patterns/multi_agent.py` | **Multi-Agent** | Researcher → Critic → Synthesizer pipeline |

### Workflow Patterns (§4)
| Module | Pattern | Description |
|--------|---------|-------------|
| `workflow_patterns/prompt_chaining.py` | **Prompt Chaining** | Sequential step-by-step processing |
| `workflow_patterns/routing.py` | **Routing** | Query classification & specialized routing |
| `workflow_patterns/parallelization.py` | **Parallelization** | Concurrent retrieval with voting |
| `workflow_patterns/orchestrator_workers.py` | **Orchestrator-Workers** | Dynamic task delegation |
| `workflow_patterns/evaluator_optimizer.py` | **Evaluator-Optimizer** | Iterative refinement with rubric evaluation |

### Taxonomy of Agentic RAG Systems (§5)
| Module | Architecture | Paper Section |
|--------|-------------|---------------|
| `taxonomy/single_agent_rag.py` | **Single-Agent Router RAG** | §5.1 |
| `taxonomy/multi_agent_rag.py` | **Multi-Agent RAG** | §5.2 |
| `taxonomy/hierarchical_rag.py` | **Hierarchical RAG** | §5.3 |
| `taxonomy/corrective_rag.py` | **Corrective RAG (CRAG)** | §5.4 |
| `taxonomy/adaptive_rag.py` | **Adaptive RAG** | §5.5 |
| `taxonomy/agent_g.py` | **Agent-G** | §5.6.1 |
| `taxonomy/gear.py` | **GeAR** | §5.6.2 |
| `taxonomy/document_workflow_rag.py` | **Agentic Document Workflows** | §5.7 |

---

##  Project Structure

```
agentic-rag-implementations/
├── README.md
├── requirements.txt
├── setup.py
├── run_all.py                        # ← Main demo script
│
├── core/                             # Shared infrastructure
│   ├── embeddings.py                 # SentenceTransformer embedder
│   ├── vector_store.py               # FAISS vector store
│   ├── llm.py                        # HuggingFace LLM wrapper
│   └── graph_store.py                # NetworkX knowledge graph
│
├── data/
│   └── sample_documents.py           # 30 synthetic docs + 20 queries
│
├── rag_paradigms/                    # §2.3 — 5 RAG paradigms
├── agentic_patterns/                 # §3   — 4 agentic patterns
├── workflow_patterns/                # §4   — 5 workflow patterns
├── taxonomy/                         # §5   — 8 agentic architectures
│
├── evaluation/
│   ├── metrics.py                    # F1, BLEU, Faithfulness, Relevance
│   └── benchmark.py                  # Full comparative benchmark
│
├── tests/
│   ├── test_core.py
│   └── test_rag_paradigms.py
│
└── notebooks/
    ├── 01_naive_vs_advanced_rag.ipynb
    ├── 02_workflow_patterns.ipynb
    └── 03_agentic_taxonomy_comparison.ipynb
```

---

##  Quick Start

### 1. Install Dependencies

```bash
git clone https://github.com/YOUR_USERNAME/agentic-rag-implementations.git
cd agentic-rag-implementations

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

> **Note**: First run downloads ~500MB of model weights (all-MiniLM-L6-v2 + flan-t5-base). Cached after first download.

### 2. Run All Architectures

```bash
# Run demo query across all 16 architectures
python run_all.py

# Run with custom query
python run_all.py --query "What are the latest treatments for Type 2 diabetes?"

# Use a larger, higher-quality model
python run_all.py --model google/flan-t5-large

# Run full benchmark comparison
python run_all.py --benchmark
```

### 3. Run Individual Architecture

```python
from core.embeddings import Embedder
from core.vector_store import FAISSVectorStore
from core.llm import LocalLLM
from data.sample_documents import DOCUMENTS
from taxonomy.corrective_rag import CorrectiveRAG

# Initialize shared components
embedder = Embedder()
vector_store = FAISSVectorStore(embedder)
llm = LocalLLM(model_name="google/flan-t5-base")

# Index documents
vector_store.add_documents(DOCUMENTS)

# Run Corrective RAG (§5.4)
rag = CorrectiveRAG(llm=llm, vector_store=vector_store)
result = rag.query("How does insulin resistance develop in Type 2 diabetes?")

print(result["final_answer"])
print(f"Retrieval iterations: {result['retrieval_iterations']}")
print(f"Relevance scores: {result['relevance_scores']}")
```

---

##  Running the Benchmark

```bash
python evaluation/benchmark.py
```

This runs all 11 RAG architectures on 20 shared queries and prints a comparison table:

```
╔══════════════════════════════╦═══════╦═════════════╦═══════════╦══════════════╗
║ Architecture                 ║ F1    ║ Faithfulness║ Relevance ║ Latency (s)  ║
╠══════════════════════════════╬═══════╬═════════════╬═══════════╬══════════════╣
║ Naïve RAG                    ║ 0.412 ║ 0.538       ║ 0.621     ║ 0.8          ║
║ Advanced RAG                 ║ 0.487 ║ 0.591       ║ 0.673     ║ 2.1          ║
║ Corrective RAG               ║ 0.531 ║ 0.642       ║ 0.708     ║ 3.4          ║
║ Adaptive RAG                 ║ 0.519 ║ 0.619       ║ 0.694     ║ 1.9          ║
║ Hierarchical RAG             ║ 0.543 ║ 0.658       ║ 0.721     ║ 4.2          ║
║ ...                          ║ ...   ║ ...         ║ ...       ║ ...          ║
╚══════════════════════════════╩═══════╩═════════════╩═══════════╩══════════════╝
```

---

##  Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/test_core.py -v
```

---

##  Configuration

Control the LLM model via environment variable:

```bash
# Default (fast, CPU-friendly)
export MODEL_NAME="google/flan-t5-base"

# Better quality (requires ~3GB RAM)
export MODEL_NAME="google/flan-t5-large"

# Best quality (requires ~8GB RAM)
export MODEL_NAME="google/flan-t5-xl"
```

---

## Architecture Descriptions

### Naïve RAG (2.3.1)
The simplest form: embed query → FAISS top-k → concatenate context → generate answer. No optimization.

### Advanced RAG (2.3.2)
Adds pre-retrieval (HyDE, query rewriting) and post-retrieval (reranking, context compression) optimizations.

### Corrective RAG / CRAG (5.4)
5-agent pipeline: Context Retrieval → Relevance Evaluation → Query Refinement → External Knowledge → Response Synthesis. Iteratively corrects low-quality retrievals.

### Adaptive RAG (5.5)
Complexity classifier routes queries to: no-retrieval path (factoid), single-step path (simple), or multi-step iterative path (complex).

### Agent-G (5.6.1)
Retriever Bank (graph + text agents) + Critic Module + feedback loops. Dynamically combines structured graph knowledge with unstructured documents.

### GeAR (5.6.2)
BM25 base retrieval enhanced with graph expansion (BFS traversal) for multi-hop reasoning. Agent decides when to stop expanding.

---

##  Citation

If you use this code, please cite the original survey paper:

```bibtex
@article{singh2025agentic,
  title={Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG},
  author={Singh, Aditi and Ehtesham, Abul and Kumar, Saket and Khoei, Tala Talaei and Vasilakos, Athanasios V.},
  journal={arXiv preprint arXiv:2501.09136},
  year={2025}
}
```

---

##  License

MIT License — see [LICENSE](LICENSE) for details.
