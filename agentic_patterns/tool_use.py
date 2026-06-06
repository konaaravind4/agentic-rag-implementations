"""
Tool Use Pattern — §3.3 of 'Agentic RAG: A Survey' (arXiv:2501.09136)
=======================================================================
Implements a tool-augmented agent that dynamically selects and invokes
specialised tools to answer a query before synthesising a final response.

Pre-built tools provided:
- ``vector_search_tool``   — dense retrieval via FAISS
- ``calculator_tool``      — safe evaluation of arithmetic expressions
- ``keyword_search_tool``  — BM25-style keyword matching over a document list
- ``summarize_tool``       — abstractive summarisation of a supplied text

References
----------
- Schick et al., "Toolformer: Language Models Can Teach Themselves to Use Tools", NeurIPS 2023.
- Survey §3.3: Tool use as a core agentic design pattern.
"""

from __future__ import annotations

import ast
import json
import operator
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from core.llm import LocalLLM
    from core.vector_store import FAISSVectorStore


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_TOOL_SELECT_PROMPT = """You are an AI assistant that selects the most appropriate tool to answer a query.

Available tools:
{tool_descriptions}

User query: {query}

Choose the SINGLE best tool for this query.
Respond ONLY with a JSON object in this exact format:
{{
  "tool_name": "<exact tool name from the list>",
  "reason": "<brief reason for selection>"
}}"""

_FINAL_ANSWER_PROMPT = """You are a helpful assistant. A tool was used to gather information to answer the user's query.

User Query: {query}

Tool Used: {tool_name}
Tool Result:
{tool_result}

Using the tool result, write a clear and comprehensive final answer to the user's query.

Final Answer:"""


# ---------------------------------------------------------------------------
# Tool dataclass
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    """
    Represents a callable tool available to the agent.

    Attributes
    ----------
    name : str
        Unique identifier for the tool.
    description : str
        Human-readable description used by the LLM for tool selection.
    fn : Callable[[str], str]
        The underlying function.  Receives the query string and returns
        a string result.
    """

    name: str
    description: str
    fn: Callable[[str], str]


# ---------------------------------------------------------------------------
# Pre-built tool factory functions
# ---------------------------------------------------------------------------

def _make_vector_search_tool(vector_store: "FAISSVectorStore", top_k: int = 5) -> Tool:
    """
    Create a vector search tool backed by a FAISS vector store.

    Parameters
    ----------
    vector_store : FAISSVectorStore
        The vector store to search.
    top_k : int, optional
        Number of results to retrieve (default 5).

    Returns
    -------
    Tool
        A ``Tool`` instance wrapping FAISS similarity search.
    """

    def _fn(query: str) -> str:
        docs = vector_store.search(query, top_k=top_k)
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
        return "\n\n---\n\n".join(chunks) if chunks else "No results found."

    return Tool(
        name="vector_search_tool",
        description=(
            "Performs dense semantic vector search over the document store. "
            "Best for general knowledge questions, conceptual queries, and "
            "finding relevant paragraphs or sections in documents."
        ),
        fn=_fn,
    )


def _make_calculator_tool() -> Tool:
    """
    Create a safe arithmetic calculator tool using ``ast``-based evaluation.

    Supports: +, -, *, /, //, %, ** and integer/float literals.
    Does NOT execute arbitrary Python code.

    Returns
    -------
    Tool
        A ``Tool`` instance for safe arithmetic computation.
    """

    # Whitelist of permitted AST node types
    _ALLOWED_NODES = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Num,         # Python < 3.8
        ast.Constant,    # Python >= 3.8
        ast.Add, ast.Sub, ast.Mult, ast.Div,
        ast.FloorDiv, ast.Mod, ast.Pow,
        ast.USub, ast.UAdd,
    )

    _OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _eval_node(node):
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value)}")
        elif isinstance(node, ast.Num):  # Python < 3.8 compatibility
            return node.n
        elif isinstance(node, ast.BinOp):
            op_fn = _OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported binary operator: {type(node.op)}")
            return op_fn(_eval_node(node.left), _eval_node(node.right))
        elif isinstance(node, ast.UnaryOp):
            op_fn = _OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op)}")
            return op_fn(_eval_node(node.operand))
        else:
            raise ValueError(f"Unsupported AST node: {type(node)}")

    def _fn(query: str) -> str:
        # Extract the math expression from the query string
        expr_match = re.search(
            r"[\d\+\-\*\/\(\)\.\s\%\^]+",
            query.replace("^", "**"),  # normalise caret exponentiation
        )
        if not expr_match:
            return "Could not extract a mathematical expression from the query."
        expr = expr_match.group().strip()
        try:
            tree = ast.parse(expr, mode="eval")
            # Validate that only allowed nodes are present
            for node in ast.walk(tree):
                if not isinstance(node, _ALLOWED_NODES):
                    raise ValueError(f"Disallowed AST node: {type(node)}")
            result = _eval_node(tree)
            return f"Result: {expr} = {result}"
        except ZeroDivisionError:
            return "Error: Division by zero."
        except Exception as exc:
            return f"Calculation error: {exc}"

    return Tool(
        name="calculator_tool",
        description=(
            "Safely evaluates arithmetic and mathematical expressions. "
            "Best for numerical computations, unit conversions, or any query "
            "requiring exact calculation (e.g. '23 * 47', '(10 + 5) / 3')."
        ),
        fn=_fn,
    )


def _make_keyword_search_tool(documents: list[str]) -> Tool:
    """
    Create a BM25-style keyword search tool over a fixed document list.

    Uses TF-IDF approximation (term frequency × inverse document frequency)
    implemented with pure Python for zero-dependency portability.

    Parameters
    ----------
    documents : list[str]
        List of document strings to search over.

    Returns
    -------
    Tool
        A ``Tool`` instance for keyword-based search.
    """
    import math

    def _tokenise(text: str) -> list[str]:
        """Lowercase, strip punctuation, split on whitespace."""
        return re.findall(r"\b[a-z]{2,}\b", text.lower())

    def _build_idf(docs: list[str]) -> dict[str, float]:
        """Compute IDF for each term in the corpus."""
        n_docs = len(docs)
        df: Counter = Counter()
        for doc in docs:
            terms = set(_tokenise(doc))
            df.update(terms)
        return {
            term: math.log((n_docs + 1) / (count + 1)) + 1.0
            for term, count in df.items()
        }

    # Pre-compute IDF at construction time
    idf = _build_idf(documents) if documents else {}

    def _score(doc: str, query_terms: list[str]) -> float:
        """Compute TF-IDF score for a single document."""
        tokens = _tokenise(doc)
        if not tokens:
            return 0.0
        tf: Counter = Counter(tokens)
        total = len(tokens)
        score = sum(
            (tf[t] / total) * idf.get(t, 1.0) for t in query_terms if t in tf
        )
        return score

    def _fn(query: str) -> str:
        if not documents:
            return "No documents available for keyword search."
        query_terms = _tokenise(query)
        if not query_terms:
            return "No valid search terms found in query."
        scored = sorted(
            enumerate(documents),
            key=lambda x: _score(x[1], query_terms),
            reverse=True,
        )
        top_docs = [doc for _, doc in scored[:3]]
        if not any(_score(doc, query_terms) > 0 for doc in top_docs):
            return "No relevant documents found for the keyword search."
        return "\n\n---\n\n".join(top_docs)

    return Tool(
        name="keyword_search_tool",
        description=(
            "Performs BM25-style keyword/term frequency search over a document "
            "corpus. Best for exact term matching, specific named-entity lookups, "
            "or when dense retrieval may miss precise keyword matches."
        ),
        fn=_fn,
    )


def _make_summarize_tool(llm: "LocalLLM") -> Tool:
    """
    Create a summarisation tool powered by the LLM.

    The tool expects the query to contain the text to summarise (either
    directly or extracted via a simple heuristic).

    Parameters
    ----------
    llm : LocalLLM
        The language model used for abstractive summarisation.

    Returns
    -------
    Tool
        A ``Tool`` instance for text summarisation.
    """
    _SUMMARIZE_PROMPT = (
        "Please provide a concise, clear summary of the following text in "
        "3–5 sentences, capturing the main points:\n\n{text}\n\nSummary:"
    )

    def _fn(query: str) -> str:
        # Use the full query as the text to summarise
        prompt = _SUMMARIZE_PROMPT.format(text=query)
        return llm.generate(prompt).strip()

    return Tool(
        name="summarize_tool",
        description=(
            "Summarises a provided piece of text into a concise overview. "
            "Best for condensing long documents, extracting key points, or "
            "when the query itself contains a large body of text to be shortened."
        ),
        fn=_fn,
    )


# ---------------------------------------------------------------------------
# ToolUseAgent
# ---------------------------------------------------------------------------

class ToolUseAgent:
    """
    Implements the Tool Use agentic design pattern for RAG (§3.3).

    The agent maintains a registry of tools, uses the LLM to select the
    most appropriate tool for each query, executes it, and synthesises
    a final answer from the tool output.

    Parameters
    ----------
    llm : LocalLLM
        Language model used for tool selection and answer synthesis.
    vector_store : FAISSVectorStore
        FAISS vector store (used by the built-in vector search tool).
    documents : list[str], optional
        Raw document list for the keyword search tool.
    top_k : int, optional
        Number of documents for vector search (default 5).
    """

    def __init__(
        self,
        llm: "LocalLLM",
        vector_store: "FAISSVectorStore",
        documents: list[str] | None = None,
        top_k: int = 5,
    ) -> None:
        self.llm = llm
        self.vector_store = vector_store
        self._tools: dict[str, Tool] = {}

        # Register the four pre-built tools
        self.register_tool(_make_vector_search_tool(vector_store, top_k))
        self.register_tool(_make_calculator_tool())
        self.register_tool(_make_keyword_search_tool(documents or []))
        self.register_tool(_make_summarize_tool(llm))

    # ------------------------------------------------------------------
    # Tool registry
    # ------------------------------------------------------------------

    def register_tool(self, tool: Tool) -> None:
        """
        Add a tool to the agent's registry.

        Parameters
        ----------
        tool : Tool
            The tool to register.  If a tool with the same name already
            exists it will be overwritten.
        """
        self._tools[tool.name] = tool

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _select_tool(self, query: str) -> Tool:
        """
        Use the LLM to select the most appropriate tool for the query.

        Builds a prompt listing all registered tools and their descriptions,
        then parses the LLM response for a ``tool_name`` JSON field.
        Falls back to ``vector_search_tool`` if parsing fails.

        Parameters
        ----------
        query : str
            The user question.

        Returns
        -------
        Tool
            The selected tool.
        """
        tool_descriptions = "\n".join(
            f"- {tool.name}: {tool.description}"
            for tool in self._tools.values()
        )
        prompt = _TOOL_SELECT_PROMPT.format(
            tool_descriptions=tool_descriptions, query=query
        )
        raw = self.llm.generate(prompt).strip()

        # Parse JSON response
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                tool_name = data.get("tool_name", "").strip()
                if tool_name in self._tools:
                    return self._tools[tool_name]
            except json.JSONDecodeError:
                pass

        # Fallback: search for a known tool name in the response
        for name in self._tools:
            if name in raw:
                return self._tools[name]

        # Ultimate fallback
        return self._tools.get("vector_search_tool", next(iter(self._tools.values())))

    def _execute_tool(self, tool: Tool, query: str) -> str:
        """
        Invoke the selected tool with the query and return its output.

        Parameters
        ----------
        tool : Tool
            The tool to execute.
        query : str
            The original user query passed as the tool's input.

        Returns
        -------
        str
            Raw string output from the tool function.
        """
        try:
            return tool.fn(query)
        except Exception as exc:
            return f"Tool '{tool.name}' encountered an error: {exc}"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def query(self, question: str) -> dict:
        """
        Answer a question using the tool use pattern.

        Workflow
        --------
        1. LLM selects the best tool from the registry.
        2. The selected tool is invoked with the question.
        3. The LLM synthesises a final answer from the tool output.

        Parameters
        ----------
        question : str
            The user question.

        Returns
        -------
        dict
            {
              "tool_used"    : str,   # name of the selected tool
              "tool_result"  : str,   # raw tool output
              "final_answer" : str,   # synthesised final answer
              "latency"      : float, # wall-clock time in seconds
            }
        """
        t_start = time.perf_counter()

        # Step 1: Select tool
        selected_tool = self._select_tool(question)

        # Step 2: Execute tool
        tool_result = self._execute_tool(selected_tool, question)

        # Step 3: Synthesise final answer
        synthesis_prompt = _FINAL_ANSWER_PROMPT.format(
            query=question,
            tool_name=selected_tool.name,
            tool_result=tool_result,
        )
        final_answer = self.llm.generate(synthesis_prompt).strip()

        latency = time.perf_counter() - t_start

        return {
            "tool_used": selected_tool.name,
            "tool_result": tool_result,
            "final_answer": final_answer,
            "latency": round(latency, 4),
        }
