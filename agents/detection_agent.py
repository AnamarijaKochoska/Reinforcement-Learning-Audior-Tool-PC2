"""
agents/detection_agent.py
-------------------------
Agent that runs LLM-based practice detection.

Owns the tools on detection_server:
  run_detection, run_detection_on_files, get_findings, get_detection_status

Keeping this agent separate from ReportAgent means adding future
capabilities (e.g. "suggest a fix for each non-supported file") belongs
to a new agent, not to this one.
"""

from __future__ import annotations
from typing import Any, Dict, List

from agents.base_agent import BaseAgent
from memory.short_term import ScanContext


class DetectionAgent(BaseAgent):
    agent_id   = "detection_agent"
    agent_type = "detection"

    def run_detection(
        self,
        ctx: ScanContext,
        practices: List[str] | None = None,
    ) -> Dict[str, Any]:
        self._mark_running(f"detection:scan={ctx.scan_run_id}")
        resp = self._call(
            "detection_server", "run_detection",
            {"scan_run_id": ctx.scan_run_id,
             "practices":   practices},
        )
        if resp.is_error:
            self._mark_error(resp.get_text())
            raise RuntimeError(f"detection failed: {resp.get_text()}")
        data = resp.get_data()
        ctx.findings = self.db.get_findings(ctx.scan_run_id)
        ctx.note(
            f"detection: {data.get('total_pairs', 0)} pairs processed, "
            f"{data.get('warnings', 0)} warnings"
        )
        self._log(
            f"[DetectionAgent] detection state: {data.get('state')} — "
            f"{data.get('total_pairs', 0)} pair(s), "
            f"{data.get('warnings', 0)} parse warning(s)"
        )
        self._mark_complete()
        return data

    def get_findings(self, ctx: ScanContext) -> Dict[str, Any]:
        resp = self._call(
            "detection_server", "get_findings",
            {"scan_run_id": ctx.scan_run_id},
        )
        return resp.get_data()
