"""
detectors/sim_async_parallel_conversation.py
--------------------------------------------
Builds the few-shot Conversation for detecting
Simulation-Based (Async/Parallel) data collection in RL code.

Prompt files (all under prompts/sim_async_parallel/):
  system.txt          – LLM role / output format rules
  question.txt        – The detection question
  sample_context.txt  – A positive example file (few-shot input)
  sample_answer.txt   – The correct response for the example (few-shot output)

The few-shot pair appears ONCE.  The target file is appended as the
final user message so the LLM sees: system → example Q → example A → real Q.
"""

from __future__ import annotations
from pathlib import Path
import sys

# Make sure the project root is importable regardless of how the script is run
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.llm import Conversation

_PROMPT_DIR = _PROJECT_ROOT / "prompts" / "sim_async_parallel"


def _read(filename: str) -> str:
    path = _PROMPT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}\n"
            f"Expected directory layout:\n"
            f"  prompts/sim_async_parallel/system.txt\n"
            f"  prompts/sim_async_parallel/question.txt\n"
            f"  prompts/sim_async_parallel/sample_context.txt\n"
            f"  prompts/sim_async_parallel/sample_answer.txt"
        )
    return path.read_text(encoding="utf-8")


# Load prompts at import time so missing files are caught early
_SYSTEM    = _read("system.txt")
_QUESTION  = _read("question.txt")
_SAMPLE_CTX = _read("sample_context.txt")
_SAMPLE_ANS = _read("sample_answer.txt")


def create_python_rl_sim_async_parallel_conversation(
    target_context: str,
) -> Conversation:
    """
    Build a few-shot Conversation for the LLM.

    Parameters
    ----------
    target_context : str
        The file content to analyse, already formatted with line numbers
        and wrapped in a ```python``` fence by the orchestrator.

    Returns
    -------
    Conversation
        Ready to pass directly to OllamaClient.chat().
    """
    return Conversation(
        messages=[
            # ── System role ────────────────────────────────────────────────
            {"role": "system", "content": _SYSTEM},

            # ── Few-shot example (ONE pair is enough) ─────────────────────
            {
                "role": "user",
                "content": _QUESTION + "\n\n" + _SAMPLE_CTX,
            },
            {
                "role": "assistant",
                "content": _SAMPLE_ANS,
            },

            # ── Real target ───────────────────────────────────────────────
            {
                "role": "user",
                "content": _QUESTION + "\n\n" + target_context,
            },
        ]
    )
