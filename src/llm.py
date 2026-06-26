from __future__ import annotations
from typing import Dict, List, Iterator
import requests
import os


DETECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "supported": {"type": "boolean"},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "line_number":  {"type": "integer"},
                    "code_snippet": {"type": "string"},
                    "explanation":  {"type": "string"},
                },
                "required": ["code_snippet", "explanation"],
            },
        },
        "assets": {"type": "object"},
    },
    "required": ["supported", "evidence"],
}



class Conversation:
    """Ordered list of role/content message dicts, as expected by Ollama /api/chat."""

    def __init__(self, messages: List[Dict[str, str]]):
        if not messages:
            raise ValueError("Conversation must contain at least one message.")
        self.messages = messages

    def __iter__(self) -> Iterator[Dict[str, str]]:
        return iter(self.messages)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Conversation(messages={len(self.messages)})"


class OllamaClient:
    """
    Thin wrapper around the Ollama /api/chat endpoint.

    Parameters
    ----------
    base_url : str
        E.g. "http://127.0.0.1:11434"
    model : str
        E.g. "llama3.1:8b"
    max_tokens : int
        Maps to Ollama's num_predict option.
    timeout : int
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        model: str = "llama3.1:8b",
        max_tokens: int = 600,
        timeout: int = 180,
        force_json: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.force_json = force_json

    def chat(self, conversation: Conversation) -> str:
        """Send a Conversation to Ollama and return the raw text response."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": conversation.messages,
            "stream": False,
            "options": {"num_predict": self.max_tokens},
        }
        if self.force_json:
            payload["format"] = DETECTION_SCHEMA
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.base_url}. "
                "Is it running? Try: ollama serve"
            )
        return response.json()["message"]["content"]