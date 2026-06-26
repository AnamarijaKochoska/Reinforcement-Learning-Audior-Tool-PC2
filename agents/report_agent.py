"""
agents/report_agent.py
----------------------
Agent that turns findings into JSON/HTML reports. Backed by
report_generator_server — no LLM involvement.

Can be invoked standalone against an existing scan_run_id to re-generate
a report without re-running detection. This is the canonical way to
"just produce the complete report" for a scan from any checkpoint.
"""

from __future__ import annotations
from typing import Any, Dict

from agents.base_agent import BaseAgent
from memory.short_term import ScanContext


class ReportAgent(BaseAgent):
    agent_id   = "report_agent"
    agent_type = "report"

    def generate_report(
        self,
        ctx: ScanContext,
        format: str = "all",
        stem: str = "scan",
    ) -> Dict[str, Any]:
        self._mark_running(f"report:scan={ctx.scan_run_id}:{format}")
        resp = self._call(
            "report_generator_server", "generate_report",
            {"scan_run_id": ctx.scan_run_id,
             "format":      format,
             "stem":        stem},
        )
        if resp.is_error:
            self._mark_error(resp.get_text())
            raise RuntimeError(f"report generation failed: {resp.get_text()}")
        data = resp.get_data()
        ctx.report_paths = data.get("report_paths", {})
        ctx.note(f"report: wrote {list(ctx.report_paths.keys())}")
        self._log(f"[ReportAgent] Wrote reports: {ctx.report_paths}")
        self._mark_complete()
        return data
