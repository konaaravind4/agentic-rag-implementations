"""
core/llm.py
===========
Local LLM wrapper for the Agentic RAG pipeline.

Supports encoder-decoder models (e.g. ``google/flan-t5-base``) and
causal / decoder-only models (e.g. ``gpt2``, ``EleutherAI/gpt-neo-*``).
The architecture is detected automatically; seq2seq models are loaded via
``AutoModelForSeq2SeqLM`` + ``AutoTokenizer`` (the ``text2text-generation``
pipeline task was removed in newer transformers releases), while causal
models continue to use the ``text-generation`` pipeline.

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

# Causal pipeline task (still supported in all transformers versions)
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


def _is_seq2seq(model_name: str) -> bool:
    """Return True if the model identifier looks like a seq2seq architecture."""
    lower = model_name.lower()
    return any(prefix in lower for prefix in _SEQ2SEQ_PREFIXES)


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
        _seq2seq (bool): True when the model is an encoder-decoder.
        _model: Loaded model object (None until first use).
        _tokenizer: Loaded tokenizer (None until first use).
        _pipeline: Loaded causal pipeline (None until first use, seq2seq=False only).

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
        self._seq2seq: bool = _is_seq2seq(model_name)
        self._model = None       # AutoModelForSeq2SeqLM  (seq2seq only)
        self._tokenizer = None   # AutoTokenizer          (seq2seq only)
        self._pipeline = None    # text-generation pipeline (causal only)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_pipeline(self) -> None:
        """
        Load the model into memory if not yet loaded.

        For seq2seq models: loads ``AutoTokenizer`` + ``AutoModelForSeq2SeqLM``.
        For causal models:  loads a HuggingFace ``text-generation`` pipeline.

        This method is idempotent; multiple calls are safe.
        """
        if self._seq2seq:
            if self._model is not None:
                return
            self._load_seq2seq()
        else:
            if self._pipeline is not None:
                return
            self._load_causal()

    def _load_seq2seq(self) -> None:
        """Load AutoTokenizer + AutoModelForSeq2SeqLM for encoder-decoder models."""
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required. "
                "Install with: pip install transformers torch"
            ) from exc

        logger.info("Loading seq2seq model '%s' …", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        if self.device != "cpu":
            self._model = self._model.to(self.device)

        self._model.eval()
        logger.info("Seq2seq model loaded successfully.")

    def _load_causal(self) -> None:
        """Load a text-generation pipeline for decoder-only models."""
        try:
            from transformers import pipeline  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "transformers is required. "
                "Install it with: pip install transformers"
            ) from exc

        device_id = -1
        if self.device.startswith("cuda"):
            parts = self.device.split(":")
            device_id = int(parts[1]) if len(parts) > 1 else 0

        logger.info(
            "Loading causal model '%s' on device=%d …", self.model_name, device_id
        )
        self._pipeline = pipeline(
            task=_CAUSAL_TASK,
            model=self.model_name,
            device=device_id,
        )
        logger.info("Causal model loaded successfully.")

    @staticmethod
    def _extract_text(output) -> str:
        """Extract generated text string from a HuggingFace pipeline output."""
        if not output:
            return ""
        first = output[0] if isinstance(output, list) else output
        text = first.get("generated_text", "")
        return text.strip()

    def _build_relevance_prompt(self, query: str, document: str) -> str:
        """Build an instruction prompt that asks the model to judge relevance."""
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

        For seq2seq models (flan-t5 etc.) the prompt is the source sequence
        and the model produces a target sequence.  For causal models the
        prompt is prepended to the generated continuation.

        Args:
            prompt (str): The input / instruction text.
            max_new_tokens (int): Maximum number of new tokens to generate.
                Defaults to ``256``.
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
