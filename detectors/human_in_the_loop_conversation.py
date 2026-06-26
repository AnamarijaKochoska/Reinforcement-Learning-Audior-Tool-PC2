"""
detectors/human_in_the_loop_conversation.py
-------------------------------------------
Builds the few-shot Conversation for detecting
Human-in-the-Loop data collection in RL code.

Prompt files (all under prompts/human_in_the_loop/):
  system.txt          - LLM role & output format rules
  question.txt        - The detection question
  sample_context.txt  - A positive few-shot example
  sample_answer.txt   - The correct response for the example

The conversation structure sent to the LLM:
  [system]    -> system.txt
  [user]      -> question.txt + sample_context.txt   (teaching example)
  [assistant] -> sample_answer.txt                   (correct answer)
  [user]      -> question.txt + <target file>        (real query)
"""

from __future__ import annotations
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.llm import Conversation

_PROMPT_DIR = _PROJECT_ROOT / "prompts" / "human_in_the_loop"


def _read(filename: str) -> str:
    path = _PROMPT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}\n"
            f"Expected files:\n"
            f"  prompts/human_in_the_loop/system.txt\n"
            f"  prompts/human_in_the_loop/question.txt\n"
            f"  prompts/human_in_the_loop/sample_context.txt\n"
            f"  prompts/human_in_the_loop/sample_answer.txt"
        )
    return path.read_text(encoding="utf-8")


_SYSTEM     = _read("system.txt")
_QUESTION   = _read("question.txt")
_SAMPLE_CTX = _read("sample_context.txt")
_SAMPLE_ANS = _read("sample_answer.txt")


def create_python_rl_human_in_the_loop_conversation(
    target_context: str,
) -> Conversation:
    """
    Build a few-shot Conversation for the LLM to detect
    Human-in-the-Loop data collection patterns.

    Parameters
    ----------
    target_context : str
        The file content to analyse, wrapped in a fenced block with line
        numbers, as produced by the orchestrator.

    Returns
    -------
    Conversation
        Ready to pass directly to OllamaClient.chat().
    """
    return Conversation(
        messages=[
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": _QUESTION + "\n\n" + _SAMPLE_CTX,
            },
            {
                "role": "assistant",
                "content": _SAMPLE_ANS,
            },
            {
                "role": "user",
                "content": _QUESTION + "\n\n" + target_context,
            },
        ]
    )
