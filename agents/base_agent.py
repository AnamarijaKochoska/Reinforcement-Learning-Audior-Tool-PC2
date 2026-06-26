"""
agents/base_agent.py
--------------------
Minimal base class for agentic wrappers around MCP tools.

An Agent bundles (a) an identity for the agent_registry and the DB logs,
(b) a reference to the shared MCPDispatcher, and (c) a method that
orchestrates one or more tool calls and handles the short-term state
(ScanContext) updates.

The goal of having an explicit Agent layer (rather than just "functions that
call the dispatcher") is to make the multi-agent boundary a first-class
structure. Each agent gets its own row in agent_registry so external
monitoring can see which agents are running. Adding a future agent
(e.g. SuggestionAgent, TestGenerationAgent) is just adding a new subclass.
"""

from __future__ import annotations
from typing import Any, Dict, Optional

from database.db import AuditDatabase
from mcp_servers.base_server import MCPDispatcher, MCPResponse
from memory.short_term import ScanContext


class BaseAgent:
    agent_id:   str = "base_agent"
    agent_type: str = "base"

    def __init__(
        self,
        db: AuditDatabase,
        dispatcher: MCPDispatcher,
        verbose: bool = True,
    ):
        self.db = db
        self.dispatcher = dispatcher
        self.verbose = verbose
        self.db.register_agent(self.agent_id, self.agent_type)

    # ── Utilities ──────────────────────────────────────────────────────
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _mark_running(self, task: str) -> None:
        self.db.update_agent_status(self.agent_id, "running", task)

    def _mark_complete(self) -> None:
        self.db.update_agent_status(self.agent_id, "complete")

    def _mark_error(self, why: str = "") -> None:
        self.db.update_agent_status(self.agent_id, "error", why or None)

    def _call(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any] | None = None,
    ) -> MCPResponse:
        return self.dispatcher.call(server_name, tool_name, arguments or {})
