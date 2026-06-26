import os

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")

OLLAMA_MAX_TOKENS: int = int(os.getenv("OLLAMA_MAX_TOKENS", "1000"))

OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "300"))


# Maximum number of candidate .py files sent to the LLM per scan.
# 0 (or any value <= 0) means NO LIMIT — every file that passes the keyword
MAX_FILES: int = int(os.getenv("MAX_FILES", "0"))

REPORTS_DIR: str = os.getenv("REPORTS_DIR", "reports")

from pathlib import Path as _Path
DB_PATH: _Path = _Path(os.getenv("DB_PATH", str(_Path(__file__).parent / "rl_auditor.db")))
