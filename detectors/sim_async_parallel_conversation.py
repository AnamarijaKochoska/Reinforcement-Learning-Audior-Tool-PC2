from __future__ import annotations
from pathlib import Path
import sys

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

_SYSTEM    = _read("system.txt")
_QUESTION  = _read("question.txt")
_SAMPLE_CTX = _read("sample_context.txt")
_SAMPLE_ANS = _read("sample_answer.txt")


def create_python_rl_sim_async_parallel_conversation(
    target_context: str,
) -> Conversation:
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
