"""
agents/file_selection_agent.py
------------------------------
Agent responsible for the first two stages of the pipeline:
  (1) scanning the repository for source files
  (2) applying keyword-based filtering to pick candidates

It owns two MCP servers' tools through the shared dispatcher:
  repository_scanner_server.{scan_repository, get_repo_scan_summary}
  file_selection_server.{select_files, list_candidates, get_selection_status}

The split into two stages is deliberate — an external caller can invoke
just one of them ("only scan, don't select" or vice-versa).
"""

from __future__ import annotations
from typing import Any, Dict

from agents.base_agent import BaseAgent
from memory.short_term import ScanContext


class FileSelectionAgent(BaseAgent):
    agent_id   = "file_selection_agent"
    agent_type = "file_selection"

    def scan_repository(self, ctx: ScanContext) -> Dict[str, Any]:
        self._mark_running(f"repo_scan:{ctx.repo_root}")
        resp = self._call(
            "repository_scanner_server", "scan_repository",
            {"repo_root": ctx.repo_root, "scan_run_id": ctx.scan_run_id},
        )
        if resp.is_error:
            self._mark_error(resp.get_text())
            raise RuntimeError(f"repo scan failed: {resp.get_text()}")
        data = resp.get_data()
        ctx.all_source_files  = data.get("files", [])
        ctx.repo_scan_summary = data.get("summary", {})
        ctx.note(f"repo_scan: {data['summary']['total_files']} files discovered")
        self._log(
            f"[FileSelectionAgent] repo scan complete: "
            f"{data['summary']['total_files']} file(s) across "
            f"{data['summary']['by_file_type']}"
        )
        return data

    def select_files(self, ctx: ScanContext, max_files: int | None = None) -> Dict[str, Any]:
        self._mark_running(f"select_files:max={max_files if max_files else 'unlimited'}")
        resp = self._call(
            "file_selection_server", "select_files",
            {"scan_run_id": ctx.scan_run_id,
             "max_files":   max_files,
             "repo_root":   ctx.repo_root},
        )
        if resp.is_error:
            self._mark_error(resp.get_text())
            raise RuntimeError(f"file selection failed: {resp.get_text()}")
        data = resp.get_data()
        ctx.selection_summary = data.get("summary", {})
        ctx.selected_files = self.db.get_candidates_by_status(
            ctx.scan_run_id, "selected",
        )
        ctx.note(f"file_selection: {len(ctx.selected_files)} selected")
        self._log(
            f"[FileSelectionAgent] selection state: {data.get('state')} — "
            f"{len(ctx.selected_files)} candidate(s) "
            f"({'truncated' if data['summary'].get('truncated_by_max_files') else 'all'})"
        )
        self._mark_complete()
        return data
