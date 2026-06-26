"""
config.py
---------
Central configuration for the RL Data Collection Auditor (LLM edition).

All runtime settings live here. Override any value via environment variables
so the tool works in CI/CD without editing source code.
"""

import os

# ── Ollama connection ──────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# Free, open-source model via Ollama. Alternatives: "mistral:7b", "phi3:mini"
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")

# Token budget for LLM responses (keep low to avoid padding / hallucinations)
OLLAMA_MAX_TOKENS: int = int(os.getenv("OLLAMA_MAX_TOKENS", "1000"))

# HTTP timeout in seconds
OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "300"))

# ── Scan behaviour ─────────────────────────────────────────────────────────
# Maximum number of candidate .py files sent to the LLM per scan.
# 0 (or any value <= 0) means NO LIMIT — every file that passes the keyword
# filter is selected. Missing a qualifying file means missing a detection,
# so the default is now "select all".
MAX_FILES: int = int(os.getenv("MAX_FILES", "0"))

# ── Reporting ──────────────────────────────────────────────────────────────
REPORTS_DIR: str = os.getenv("REPORTS_DIR", "reports")

# ── Database ───────────────────────────────────────────────────────────────
from pathlib import Path as _Path
DB_PATH: _Path = _Path(os.getenv("DB_PATH", str(_Path(__file__).parent / "rl_auditor.db")))
