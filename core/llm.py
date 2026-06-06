"""
core/llm.py
===========
Local LLM wrapper for the Agentic RAG pipeline.

Supports encoder-decoder models (e.g. ``google/flan-t5-base``) and
causal / decoder-only models (e.g. ``gpt2``, ``EleutherAI/gpt-neo-*``).
The correct HuggingFace ``pipeline`` task type is chosen automatically based
on the model's architecture.

All inference runs locally on CPU (or GPU if available); no API keys or
network calls are required after the model is downloaded.

Dependencies:
    - transformers
    - torch
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# HuggingFace pipeline task labels
_SEQ2SEQ_TASK = "text2text-generation"
_CAUSAL_TASK = "text-generation"

# Model families that use seq2seq (encoder-decoder) architectures
_SEQ2SEQ_PREFIXES = (
    "t5",
    "flan",
    "bart",
    "pegasus",
    "marian",
    "mt5",
    "longt5",
)


def _infer_task(model_name: str) -> str:
    """
    Heuristically determine the HuggingFace pipeline task for a model.

    Args:
        model_name (str): HuggingFace model identifier or local path.

    Returns:
        str: Either ``'text2text-generation'`` or ``'text-generation'``.
    """
    lower = model_name.lower()
    for prefix in _SEQ2SEQ_PREFIXES:
        if prefix in lower:
            return _SEQ2SEQ_TASK
    return _CAUSAL_TASK


class LocalLLM:
    """
    A unified wrapper around a locally-loaded HuggingFace language model.

    Supports both seq2seq (encoder-decoder, e.g. flan-t5) and causal LM
    (decoder-only, e.g. GPT-2) architectures through a common interface.

    Model loading is lazy — no GPU/CPU memory is consumed until the first
    call to :meth:`generate` or :meth:`score_relevance`.

    Attributes:
        model_name (str): HuggingFace model identifier.
        device (str): Target device for inference (``'cpu'`` or ``'cuda'``).
        task (str): Inferred HuggingFace pipeline task.
        _pipeline: The loaded HuggingFace pipeline (``None`` until first use).

    Example:
        >>> llm = LocalLLM()
        >>> answer = llm.generate("What is the capital of France?")
        >>> print(answer)
        'Paris'
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: str = "cpu",
    ) -> None:
        """
        Initialise the LocalLLM wrapper.

        Args:
            model_name (str): HuggingFace model repository ID or local path.
                Defaults to ``'google/flan-t5-base'``.
            device (str): Device to run inference on.  Use ``'cpu'`` for
                CPU-only environments or ``'cuda'`` / ``'cuda:0'`` for GPU.
                Defaults to ``'cpu'``.
        """
        self.model_name: str = model_name
        self.device: str = device
        self.task: str = _infer_task(model_name)
        self._pipeline: Optional[object] = None  # loaded lazily

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_pipeline(self) -> None:
        """
        Load the HuggingFace pipeline into memory if not yet loaded.

        This method is idempotent; multiple calls are safe.

        Raises:
            ImportError: If ``transformers`` or ``torch`` are not installed.
            OSError: If the model cannot be downloaded or found locally.
        """
        if self._pipeline is not None:
            return

        try:
            from transformers import pipeline  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "transformers is required. "
                "Install it with: pip install transformers"
            ) from exc

        logger.info(
            "Loading model '%s' with task '%s' on device '%s' …",
            self.model_name,
            self.task,
            self.device,
        )

        # Determine the integer device index for HuggingFace pipeline
        # pipeline(device=...) accepts -1 for CPU, ≥0 for GPU
        if self.device == "cpu":
            device_id = -1
        elif self.device.startswith("cuda"):
            parts = self.device.split(":")
            device_id = int(parts[1]) if len(parts) > 1 else 0
        else:
            device_id = -1

        self._pipeline = pipeline(
            task=self.task,
            model=self.model_name,
            device=device_id,
        )
        logger.info("Model loaded successfully.")

    @staticmethod
    def _extract_text(output: list[dict]) -> str:
        """
        Pull the generated text string out of a HuggingFace pipeline output.

        Handles both seq2seq (``'generated_text'``) and causal
        (``'generated_text'``) keys, which differ only in whether the
        prompt is included in the output.

        Args:
            output (list[dict]): Raw pipeline output.

        Returns:
            str: The generated text, stripped of leading/trailing whitespace.
        """
        if not output:
            return ""
        first = output[0]
        text = first.get("generated_text", "")
        return text.strip()

    def _build_relevance_prompt(self, query: str, document: str) -> str:
        """
        Build an instruction prompt that asks the model to judge relevance.

        The model is asked to output a single floating-point number between
        0 and 1, which is then parsed in :meth:`score_relevance`.

        Args:
            query (str): The user query.
            document (str): The candidate document text (may be truncated).

        Returns:
            str: The formatted prompt string.
        """
        # Truncate document to avoid exceeding model context window
        max_doc_chars = 512
        truncated_doc = document[:max_doc_chars]
        if len(document) > max_doc_chars:
            truncated_doc += " [...]"

        return (
            f"On a scale of 0.0 to 1.0, how relevant is the following document "
            f"to the query? Output only a single number between 0.0 and 1.0.\n\n"
            f"Query: {query}\n\n"
            f"Document: {truncated_doc}\n\n"
            f"Relevance score (0.0 to 1.0):"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate a text response for the given prompt.

        For seq2seq models (flan-t5 etc.) the prompt is treated as the
        source sequence; the model produces a target sequence.  For causal
        models the prompt is prepended to the generated continuation.

        Args:
            prompt (str): The input / instruction text.
            max_new_tokens (int): Maximum number of new tokens to generate.
                Defaults to ``256``.
            temperature (float): Sampling temperature.  Lower values make
                output more deterministic; higher values introduce more
                randomness.  Defaults to ``0.7``.
                Note: flan-t5 uses greedy / beam-search by default; set
                ``temperature < 1`` to enable sampling behaviour.

        Returns:
            str: The generated text.  For causal models, the original
            prompt is stripped from the beginning of the output if present.

        Raises:
            ImportError: If ``transformers`` is not installed.

        Example:
            >>> llm = LocalLLM()
            >>> llm.generate("Summarise: Diabetes is a chronic disease ...")
            'Diabetes is a long-term condition ...'
        """
        self._load_pipeline()

        # Build generation kwargs appropriate for the pipeline task
        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
        }

        # Only enable sampling when temperature != 1.0 to avoid
        # the HuggingFace "temperature requires do_sample=True" warning
        if temperature != 1.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        logger.debug(
            "Generating with kwargs: %s | prompt_len=%d",
            gen_kwargs,
            len(prompt),
        )

        raw_output = self._pipeline(prompt, **gen_kwargs)  # type: ignore[operator]
        generated = self._extract_text(raw_output)

        # For causal LMs the output includes the prompt; strip it
        if self.task == _CAUSAL_TASK and generated.startswith(prompt):
            generated = generated[len(prompt):].strip()

        return generated

    def score_relevance(self, query: str, document: str) -> float:
        """
        Estimate the relevance of ``document`` to ``query`` using the LLM.

        The model is prompted to output a float between 0 and 1.  The raw
        generation is parsed with a regex; if no valid float is found the
        method falls back to 0.5 (neutral score).

        Args:
            query (str): The user query string.
            document (str): The candidate document text.

        Returns:
            float: A relevance score in ``[0.0, 1.0]``.  Higher is more
            relevant.

        Raises:
            ImportError: If ``transformers`` is not installed.

        Example:
            >>> llm = LocalLLM()
            >>> score = llm.score_relevance(
            ...     query="What causes type 2 diabetes?",
            ...     document="Type 2 diabetes is linked to insulin resistance ...",
            ... )
            >>> 0.0 <= score <= 1.0
            True
        """
        prompt = self._build_relevance_prompt(query, document)
        # Use greedy decoding for deterministic scoring
        raw = self.generate(prompt, max_new_tokens=16, temperature=1.0)

        # Try to extract the first float in the output
        matches = re.findall(r"\d+\.\d+|\d+", raw)
        if matches:
            try:
                score = float(matches[0])
                # Clamp to [0, 1]
                score = max(0.0, min(1.0, score))
                return score
            except ValueError:
                pass

        logger.warning(
            "Could not parse relevance score from LLM output %r; defaulting to 0.5.",
            raw,
        )
        return 0.5  # neutral fallback

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        loaded = self._pipeline is not None
        return (
            f"LocalLLM("
            f"model_name={self.model_name!r}, "
            f"device={self.device!r}, "
            f"task={self.task!r}, "
            f"loaded={loaded})"
        )
