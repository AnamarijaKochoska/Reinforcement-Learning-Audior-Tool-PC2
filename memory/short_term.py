"""
memory/short_term.py
--------------------
Short-term memory for an agentic run.

This is the in-process "working memory" — a single mutable ScanContext that
gets handed between tools in the same orchestration. It mirrors what a
LangGraph-style graph would call its "state" object: a typed dict that flows
along the edges between nodes.

The long-term store (SQLite, see database/db.py) is the source of truth and
persists across processes. The ScanContext is an ephemeral convenience so
tools in the same run don't have to re-query the DB for things they just
computed. Every field it exposes is also recoverable from the DB via
`hydrate_from_db()` — that's how resume-from-checkpoint works.

Contract:
    1. Every write to ScanContext must also be mirrored to the DB by whatever
       tool did the write (so restarts are safe).
    2. Nothing should read from ScanContext without first calling ensure_*
       helpers, which will hydrate from the DB if the field is empty.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from database.db import (
    AuditDatabase,
    STAGE_REPO_SCAN, STAGE_FILE_SELECTION, STAGE_VALIDATION,
    STAGE_DETECTION, STAGE_REPORT,
)


@dataclass
class ScanContext:
    """
    Working memory for a single scan_run. Passed through the orchestrator
    and available to any tool that needs upstream results without a DB hit.
    """
    scan_run_id: int
    repo_root: str
    model: str

    # Stage 1 (repo scan) output
    all_source_files:     List[str]      = field(default_factory=list)
    repo_scan_summary:    Dict[str, Any] = field(default_factory=dict)

    # Stage 2 (file selection) output
    selected_files:       List[Dict[str, Any]] = field(default_factory=list)
    selection_summary:    Dict[str, Any]       = field(default_factory=dict)

    # Stage 3 (validation) output
    validated_files:      List[Dict[str, Any]] = field(default_factory=list)
    rejected_files:       List[Dict[str, Any]] = field(default_factory=list)

    # Stage 4 (detection) output
    findings:             List[Dict[str, Any]] = field(default_factory=list)

    # Stage 5 (report) output
    report_paths:         Dict[str, str] = field(default_factory=dict)

    # Free-form breadcrumbs for debugging
    notes:                List[str] = field(default_factory=list)

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    # ── Hydration helpers ──────────────────────────────────────────────
    # Each helper refills a field from the DB if it's currently empty.
    # This is how a tool invoked standalone ("run report from checkpoint")
    # recovers everything it needs without re-running upstream stages.

    def ensure_selected(self, db: AuditDatabase) -> List[Dict[str, Any]]:
        if not self.selected_files:
            self.selected_files = db.get_candidates_by_status(
                self.scan_run_id, "selected"
            )
            # If the selection has already been consumed by validation,
            # also consider files that moved on to 'validated'/'complete'.
            if not self.selected_files:
                self.selected_files = [
                    c for c in db.get_all_candidates(self.scan_run_id)
                    if c["status"] in ("selected", "validated", "complete")
                ]
        return self.selected_files

    def ensure_validated(self, db: AuditDatabase) -> List[Dict[str, Any]]:
        if not self.validated_files:
            self.validated_files = db.get_candidates_by_status(
                self.scan_run_id, "validated"
            )
            # Files that have already moved on to 'complete' were also
            # validated — include them for downstream re-runs.
            self.validated_files += db.get_candidates_by_status(
                self.scan_run_id, "complete"
            )
        return self.validated_files

    def ensure_findings(self, db: AuditDatabase) -> List[Dict[str, Any]]:
        if not self.findings:
            self.findings = db.get_findings(self.scan_run_id)
        return self.findings

    # ── Checkpoint round-trip ──────────────────────────────────────────
    def to_checkpoint_payload(self, stage: str) -> Dict[str, Any]:
        """Serialize the slice of context relevant to a given stage."""
        if stage == STAGE_REPO_SCAN:
            return {
                "all_source_files":  self.all_source_files,
                "repo_scan_summary": self.repo_scan_summary,
            }
        if stage == STAGE_FILE_SELECTION:
            return {
                "selected_files":    [f["file_path"] for f in self.selected_files],
                "selection_summary": self.selection_summary,
            }
        if stage == STAGE_VALIDATION:
            return {
                "validated_files":   [f["file_path"] for f in self.validated_files],
                "rejected_files":    [f["file_path"] for f in self.rejected_files],
            }
        if stage == STAGE_DETECTION:
            return {
                "findings_count": len(self.findings),
                "files_analyzed": list({f["file_path"] for f in self.findings}),
            }
        if stage == STAGE_REPORT:
            return {"report_paths": self.report_paths}
        return {}

    @classmethod
    def hydrate_from_db(
        cls, db: AuditDatabase, scan_run_id: int,
    ) -> "ScanContext":
        """
        Rebuild a ScanContext from persistent storage. Used when a tool is
        invoked standalone for an existing scan run — e.g. generating a
        report for scan 5 without re-running selection/detection.
        """
        run = db.get_scan_run(scan_run_id)
        if run is None:
            raise ValueError(f"No scan run with id={scan_run_id}")
        ctx = cls(
            scan_run_id=scan_run_id,
            repo_root=run["repo_root"],
            model=run["model"] or "",
        )
        ctx.ensure_selected(db)
        ctx.ensure_validated(db)
        ctx.ensure_findings(db)

        # Also try to pick up checkpointed data for the non-candidate fields.
        for stage in (STAGE_REPO_SCAN, STAGE_FILE_SELECTION,
                      STAGE_VALIDATION, STAGE_DETECTION, STAGE_REPORT):
            cp = db.load_checkpoint(scan_run_id, stage)
            if not cp:
                continue
            p = cp["payload"]
            if stage == STAGE_REPO_SCAN:
                ctx.all_source_files  = p.get("all_source_files", [])
                ctx.repo_scan_summary = p.get("repo_scan_summary", {})
            elif stage == STAGE_FILE_SELECTION:
                ctx.selection_summary = p.get("selection_summary", {})
            elif stage == STAGE_REPORT:
                ctx.report_paths = p.get("report_paths", {})
        return ctx
