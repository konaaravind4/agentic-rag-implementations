"""
Multi-Agent Collaboration — §3.4 of 'Agentic RAG: A Survey' (arXiv:2501.09136)
================================================================================
Implements a lightweight multi-agent system in which specialised agents
communicate through a message-passing orchestrator.

Pipeline (Research → Critique → Synthesis):
  1. ``ResearcherAgent``   — retrieves and summarises relevant documents.
  2. ``CriticAgent``       — evaluates the researcher's output for quality issues.
  3. ``SynthesizerAgent``  — combines the research and critique into a polished answer.

The ``MultiAgentOrchestrator`` coordinates message routing and maintains a
shared message log for full pipeline transparency.

References
----------
- Park et al., "Generative Agents", UIST 2023.
- Survey §3.4: Multi-agent collaboration as an agentic design pattern.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.llm import LocalLLM
    from core.vector_store import FAISSVectorStore


# ---------------------------------------------------------------------------
# Message dataclass
# ---------------------------------------------------------------------------

@dataclass
class AgentMessage:
    """
    Represents a message passed between agents.

    Attributes
    ----------
    sender : str
        Name of the sending agent.
    recipient : str
        Name of the intended receiving agent.
    content : str
        The message payload (text).
    message_type : str
        Semantic label for the message, e.g. ``'query'``, ``'research'``,
        ``'critique'``, ``'synthesis'``.
    metadata : dict
        Optional key-value metadata attached to the message.
    """

    sender: str
    recipient: str
    content: str
    message_type: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract BaseAgent
# ---------------------------------------------------------------------------

class BaseAgent(abc.ABC):
    """
    Abstract base class for all agents in the multi-agent system.

    Subclasses must implement the ``process`` method which receives an
    ``AgentMessage`` and returns an ``AgentMessage`` directed to the next
    agent (or the orchestrator).

    Attributes
    ----------
    name : str
        Unique identifier for this agent.
    role : str
        Human-readable description of the agent's purpose.
    llm : LocalLLM
        Shared language model instance.
    """

    def __init__(self, name: str, role: str, llm: "LocalLLM") -> None:
        self.name = name
        self.role = role
        self.llm = llm

    @abc.abstractmethod
    def process(self, message: AgentMessage) -> AgentMessage:
        """
        Process an incoming message and produce a reply message.

        Parameters
        ----------
        message : AgentMessage
            The incoming message to handle.

        Returns
        -------
        AgentMessage
            The outgoing response message.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# ResearcherAgent
# ---------------------------------------------------------------------------

_RESEARCHER_PROMPT = """You are a thorough research assistant. Your task is to retrieve and synthesise information to answer a question.

Question: {query}

Retrieved Context:
{context}

Based on the retrieved context, provide a detailed research summary that covers all relevant information needed to answer the question. Include key facts, concepts, and any supporting evidence.

Research Summary:"""


class ResearcherAgent(BaseAgent):
    """
    Retrieves relevant documents and produces a research summary.

    Parameters
    ----------
    llm : LocalLLM
        Language model for generating research summaries.
    vector_store : FAISSVectorStore
        Vector store for document retrieval.
    top_k : int, optional
        Number of documents to retrieve per query (default 5).
    """

    def __init__(
        self,
        llm: "LocalLLM",
        vector_store: "FAISSVectorStore",
        top_k: int = 5,
    ) -> None:
        super().__init__(
            name="ResearcherAgent",
            role="Retrieves and summarises relevant documents to answer the query.",
            llm=llm,
        )
        self.vector_store = vector_store
        self.top_k = top_k

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

    def process(self, message: AgentMessage) -> AgentMessage:
        """
        Retrieve documents and generate a research summary.

        Parameters
        ----------
        message : AgentMessage
            Expected ``message_type`` is ``'query'``; ``content`` is the question.

        Returns
        -------
        AgentMessage
            A ``'research'`` message directed to the ``CriticAgent``.
        """
        query = message.content
        context = self._retrieve_context(query)
        prompt = _RESEARCHER_PROMPT.format(query=query, context=context)
        research_summary = self.llm.generate(prompt).strip()

        return AgentMessage(
            sender=self.name,
            recipient="CriticAgent",
            content=research_summary,
            message_type="research",
            metadata={"original_query": query, "num_docs_retrieved": self.top_k},
        )


# ---------------------------------------------------------------------------
# CriticAgent
# ---------------------------------------------------------------------------

_CRITIC_PROMPT = """You are a critical evaluator. Assess the following research summary for quality and completeness.

Original Question: {query}

Research Summary:
{research}

Evaluate the research summary and provide:
1. An overall quality assessment (strengths and weaknesses).
2. Specific gaps or inaccuracies that should be addressed.
3. Suggestions for improvement.
4. A quality score from 1 to 10.

Structure your critique clearly with these sections:
- Strengths:
- Weaknesses/Gaps:
- Suggestions:
- Quality Score: X/10

Critique:"""


class CriticAgent(BaseAgent):
    """
    Evaluates the researcher's output for completeness, accuracy, and clarity.

    Parameters
    ----------
    llm : LocalLLM
        Language model used to generate critiques.
    """

    def __init__(self, llm: "LocalLLM") -> None:
        super().__init__(
            name="CriticAgent",
            role="Evaluates research quality and flags gaps or inaccuracies.",
            llm=llm,
        )

    def process(self, message: AgentMessage) -> AgentMessage:
        """
        Critique the research summary received from the ResearcherAgent.

        Parameters
        ----------
        message : AgentMessage
            Expected ``message_type`` is ``'research'``; metadata must contain
            ``'original_query'``.

        Returns
        -------
        AgentMessage
            A ``'critique'`` message directed to the ``SynthesizerAgent``.
        """
        research_summary = message.content
        query = message.metadata.get("original_query", "")

        prompt = _CRITIC_PROMPT.format(query=query, research=research_summary)
        critique = self.llm.generate(prompt).strip()

        return AgentMessage(
            sender=self.name,
            recipient="SynthesizerAgent",
            content=critique,
            message_type="critique",
            metadata={
                "original_query": query,
                "research_summary": research_summary,
            },
        )


# ---------------------------------------------------------------------------
# SynthesizerAgent
# ---------------------------------------------------------------------------

_SYNTHESIZER_PROMPT = """You are an expert synthesiser. Your task is to create a comprehensive, polished answer by combining research findings with critical feedback.

Original Question: {query}

Research Summary:
{research}

Critic's Feedback:
{critique}

Using both the research and the critic's feedback, write a comprehensive, accurate, and well-structured final answer that:
- Directly addresses the original question
- Incorporates the strengths of the research
- Addresses the weaknesses identified by the critic
- Is clear and well-organised

Final Answer:"""


class SynthesizerAgent(BaseAgent):
    """
    Synthesises the research summary and critique into a final polished answer.

    Parameters
    ----------
    llm : LocalLLM
        Language model used for synthesis.
    """

    def __init__(self, llm: "LocalLLM") -> None:
        super().__init__(
            name="SynthesizerAgent",
            role="Merges research and critique into a final comprehensive answer.",
            llm=llm,
        )

    def process(self, message: AgentMessage) -> AgentMessage:
        """
        Synthesise the research and critique into the final answer.

        Parameters
        ----------
        message : AgentMessage
            Expected ``message_type`` is ``'critique'``; metadata must contain
            ``'original_query'`` and ``'research_summary'``.

        Returns
        -------
        AgentMessage
            A ``'synthesis'`` message with the final answer.
        """
        critique = message.content
        query = message.metadata.get("original_query", "")
        research = message.metadata.get("research_summary", "")

        prompt = _SYNTHESIZER_PROMPT.format(
            query=query, research=research, critique=critique
        )
        synthesis = self.llm.generate(prompt).strip()

        return AgentMessage(
            sender=self.name,
            recipient="Orchestrator",
            content=synthesis,
            message_type="synthesis",
            metadata={"original_query": query},
        )


# ---------------------------------------------------------------------------
# MultiAgentOrchestrator
# ---------------------------------------------------------------------------

class MultiAgentOrchestrator:
    """
    Manages the multi-agent pipeline: ResearcherAgent → CriticAgent → SynthesizerAgent.

    The orchestrator handles message routing, maintains a shared message log,
    and exposes a single ``query`` interface that runs the full pipeline.

    Parameters
    ----------
    llm : LocalLLM
        Shared language model instance passed to all sub-agents.
    vector_store : FAISSVectorStore
        Vector store passed to the ResearcherAgent.
    top_k : int, optional
        Number of documents to retrieve per query (default 5).
    """

    def __init__(
        self,
        llm: "LocalLLM",
        vector_store: "FAISSVectorStore",
        top_k: int = 5,
    ) -> None:
        self._researcher = ResearcherAgent(llm, vector_store, top_k)
        self._critic = CriticAgent(llm)
        self._synthesizer = SynthesizerAgent(llm)

        # Map agent names to instances for generic routing
        self._agents: dict[str, BaseAgent] = {
            agent.name: agent
            for agent in (self._researcher, self._critic, self._synthesizer)
        }

        self.message_log: list[AgentMessage] = []

    def _route(self, message: AgentMessage) -> AgentMessage:
        """
        Route a message to the appropriate agent and return its response.

        Parameters
        ----------
        message : AgentMessage
            The message to route.

        Returns
        -------
        AgentMessage
            The response from the target agent.

        Raises
        ------
        ValueError
            If the recipient agent is not registered.
        """
        recipient_name = message.recipient
        if recipient_name not in self._agents:
            raise ValueError(
                f"Unknown agent '{recipient_name}'. "
                f"Registered agents: {list(self._agents.keys())}"
            )
        agent = self._agents[recipient_name]
        response = agent.process(message)
        self.message_log.append(message)
        return response

    def query(self, question: str) -> dict:
        """
        Run the full multi-agent pipeline for a given question.

        Pipeline
        --------
        1. Orchestrator creates an initial query message → ResearcherAgent.
        2. ResearcherAgent produces a research summary → CriticAgent.
        3. CriticAgent produces a critique → SynthesizerAgent.
        4. SynthesizerAgent produces the final synthesised answer.

        Parameters
        ----------
        question : str
            The user question to answer.

        Returns
        -------
        dict
            {
              "final_answer"   : str,        # synthesised final answer
              "message_log"    : list[dict], # serialised full message history
              "pipeline_steps" : list[str],  # ordered list of agents invoked
              "latency"        : float,      # wall-clock time in seconds
            }
        """
        t_start = time.perf_counter()
        self.message_log.clear()

        # Initial query message from the orchestrator to the researcher
        initial_message = AgentMessage(
            sender="Orchestrator",
            recipient="ResearcherAgent",
            content=question,
            message_type="query",
        )

        # Step 1: Researcher
        research_message = self._route(initial_message)
        self.message_log.append(research_message)

        # Step 2: Critic
        critique_message = self._route(research_message)
        self.message_log.append(critique_message)

        # Step 3: Synthesizer
        synthesis_message = self._route(critique_message)
        self.message_log.append(synthesis_message)

        latency = time.perf_counter() - t_start

        # Serialise message log for the return dict
        serialised_log = [
            {
                "sender": msg.sender,
                "recipient": msg.recipient,
                "message_type": msg.message_type,
                "content_preview": msg.content[:300] + ("…" if len(msg.content) > 300 else ""),
                "metadata": msg.metadata,
            }
            for msg in self.message_log
        ]

        pipeline_steps = [
            msg.sender
            for msg in self.message_log
            if msg.sender != "Orchestrator"
        ]

        return {
            "final_answer": synthesis_message.content,
            "message_log": serialised_log,
            "pipeline_steps": pipeline_steps,
            "latency": round(latency, 4),
        }
