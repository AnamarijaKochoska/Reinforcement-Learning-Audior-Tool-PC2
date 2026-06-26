from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from database.db import (
    AuditDatabase,
    STAGE_REPO_SCAN, STAGE_FILE_SELECTION, STAGE_VALIDATION,
    STAGE_DETECTION, STAGE_REPORT,
    STATE_COMPLETE, STATE_PARTIAL,
)
from mcp_servers.base_server import MCPDispatcher
from mcp_servers.repository_scanner_server import RepositoryScannerServer
from mcp_servers.file_selection_server      import FileSelectionServer
from mcp_servers.validation_server          import ValidationServer
from mcp_servers.detection_server           import DetectionServer
from mcp_servers.report_generator_server    import ReportGeneratorServer

from agents.base_agent           import BaseAgent
from agents.file_selection_agent import FileSelectionAgent
from agents.validation_agent     import ValidationAgent
from agents.detection_agent      import DetectionAgent
from agents.report_agent         import ReportAgent

from memory.short_term import ScanContext
from src.llm import OllamaClient


class OrchestratorAgent(BaseAgent):
    agent_id   = "orchestrator_agent"
    agent_type = "orchestrator"

    def __init__(
        self,
        db: AuditDatabase,
        llm_client: OllamaClient,
        detectors: Dict[str, Any],
        max_files: int | None = None,
        reports_dir: str = "reports",
        verbose: bool = True,
        max_evidence_retries: int = 2,
    ):
 
        dispatcher = MCPDispatcher()
        dispatcher.register_server(RepositoryScannerServer(db))
        dispatcher.register_server(FileSelectionServer(db))
        dispatcher.register_server(ValidationServer(db))
        dispatcher.register_server(DetectionServer(db, llm_client, detectors))
        dispatcher.register_server(ReportGeneratorServer(db, reports_dir))

        super().__init__(db=db, dispatcher=dispatcher, verbose=verbose)

        self.llm = llm_client
        self.detectors = detectors
        self.max_files = max_files
        self.reports_dir = reports_dir
        self.max_evidence_retries = max_evidence_retries

        self.file_selection_agent = FileSelectionAgent(db, dispatcher, verbose)
        self.validation_agent     = ValidationAgent(db, dispatcher, verbose)
        self.detection_agent      = DetectionAgent(db, dispatcher, verbose)
        self.report_agent         = ReportAgent(db, dispatcher, verbose)

    def list_tools(self) -> List[Dict[str, Any]]:
        return self.dispatcher.list_all_tools()

    def describe_all(self) -> List[Dict[str, Any]]:
        return self.dispatcher.describe_all()

    def _stage_already_complete(self, scan_run_id: int, stage: str) -> bool:
        row = self.db.get_stage_status(scan_run_id, stage)
        return bool(row and row["state"] == STATE_COMPLETE)

    def run(
        self,
        repo_root: str,
        scan_run_id: int | None = None,
        skip_stages: List[str] | None = None,
    ) -> Dict[str, Any]:
        
        skip = set(skip_stages or [])

        if scan_run_id is None:
            scan_run_id = self.db.create_scan_run(repo_root, self.llm.model)
            ctx = ScanContext(
                scan_run_id=scan_run_id,
                repo_root=repo_root,
                model=self.llm.model,
            )
            self._log(f"\n[Orchestrator] New scan run: id={scan_run_id}")
        else:
            ctx = ScanContext.hydrate_from_db(self.db, scan_run_id)
            self._log(
                f"\n[Orchestrator] Resuming scan run id={scan_run_id} "
                f"(repo={ctx.repo_root})"
            )

        self._mark_running(f"scan:{ctx.repo_root}")
        self._log(f"[Orchestrator] Practices: {list(self.detectors.keys())}")
        self._log(
            f"[Orchestrator] MCP servers online: "
            f"{self.dispatcher.get_server_names()}"
        )

        # ─ Stage 1: repo scan ─
        if STAGE_REPO_SCAN in skip or self._stage_already_complete(scan_run_id, STAGE_REPO_SCAN):
            self._log(f"[Orchestrator] ⇢ skipping {STAGE_REPO_SCAN} (already complete or skipped)")
        else:
            self._log(f"\n[Orchestrator] ── Stage 1: {STAGE_REPO_SCAN} ──")
            self.file_selection_agent.scan_repository(ctx)
            # Per-stage validation runs after every stage, never halts.
            self._log("[Orchestrator]    └─ validation:")
            self.validation_agent.validate_repo_scan_output(ctx)

        # ─ Stage 2: file selection ─
        if STAGE_FILE_SELECTION in skip or self._stage_already_complete(scan_run_id, STAGE_FILE_SELECTION):
            self._log(f"[Orchestrator] ⇢ skipping {STAGE_FILE_SELECTION}")
        else:
            self._log(f"\n[Orchestrator] ── Stage 2: {STAGE_FILE_SELECTION} ──")
            self.file_selection_agent.select_files(ctx, max_files=self.max_files)
            self._log("[Orchestrator]    └─ validation:")
            self.validation_agent.validate_selection_output(ctx)
        ctx.ensure_selected(self.db)

        if not ctx.selected_files:
            self._log("[Orchestrator] No candidates found. Finishing early.")
            return self._finish(ctx)

        # ─ Stage 3: file-level validation ─
        if STAGE_VALIDATION in skip or self._stage_already_complete(scan_run_id, STAGE_VALIDATION):
            self._log(f"[Orchestrator] ⇢ skipping {STAGE_VALIDATION}")
            ctx.ensure_validated(self.db)
        else:
            self._log(f"\n[Orchestrator] ── Stage 3: {STAGE_VALIDATION} ──")
            self.validation_agent.validate_files(ctx)
            self._log("[Orchestrator]    └─ validation:")
            self.validation_agent.validate_validation_output(ctx)

        if not ctx.validated_files:
            self._log("[Orchestrator] No files passed validation. Finishing early.")
            return self._finish(ctx)

        # ─ Stage 4: detection ─
        if STAGE_DETECTION in skip or self._stage_already_complete(scan_run_id, STAGE_DETECTION):
            self._log(f"[Orchestrator] ⇢ skipping {STAGE_DETECTION}")
            ctx.ensure_findings(self.db)
        else:
            self._log(f"\n[Orchestrator] ── Stage 4: {STAGE_DETECTION} ──")
            self.detection_agent.run_detection(ctx)
            self._log("[Orchestrator]    └─ validation:")
            detection_validation = self.validation_agent.validate_detection_results(ctx)
            # A finding marked supported=True MUST carry evidence. If any does
            # not, something went wrong in that (file, practice) analysis — the
            # orchestrator re-runs detection for just those pairs. If they still
            # come back empty after `max_evidence_retries`, the practice/prompts
            # are likely at fault and we leave it for human review.
            self._resolve_empty_evidence(ctx, detection_validation)

        # ─ Stage 5: report generation ─
        if STAGE_REPORT in skip:
            self._log(f"[Orchestrator] ⇢ skipping {STAGE_REPORT}")
        else:
            self._log(f"\n[Orchestrator] ── Stage 5: {STAGE_REPORT} ──")
            self.report_agent.generate_report(ctx)
            self._log("[Orchestrator]    └─ validation:")
            self.validation_agent.validate_report_output(ctx)

        return self._finish(ctx)


    def regenerate_report(self, scan_run_id: int, format: str = "all") -> Dict[str, Any]:
        ctx = ScanContext.hydrate_from_db(self.db, scan_run_id)
        return self.report_agent.generate_report(ctx, format=format)

    def rerun_detection(
        self, scan_run_id: int, practices: List[str] | None = None,
    ) -> Dict[str, Any]:
        
        ctx = ScanContext.hydrate_from_db(self.db, scan_run_id)
        if practices:
            for practice in practices:
                # Delete only findings for the practices being re-run
                with self.db._conn() as conn:
                    conn.execute(
                        "DELETE FROM findings WHERE scan_run_id=? AND practice=?",
                        (scan_run_id, practice),
                    )
        else:
            self.db.delete_all_findings(scan_run_id)
        return self.detection_agent.run_detection(ctx, practices=practices)

    def reanalyze_file(
        self,
        scan_run_id: int,
        file_path: str,
        practices: List[str] | None = None,
    ) -> Dict[str, Any]:

        resp = self.dispatcher.call(
            "detection_server", "run_detection_on_files",
            {"scan_run_id": scan_run_id,
             "file_paths":  [file_path],
             "practices":   practices},
        )
        if resp.is_error:
            raise RuntimeError(resp.get_text())
        return resp.get_data()

    def revalidate_results(self, scan_run_id: int) -> Dict[str, Any]:
        ctx = ScanContext.hydrate_from_db(self.db, scan_run_id)
        return self.validation_agent.validate_results(ctx)

    def rerun_stage(
        self, scan_run_id: int, stage: str,
    ) -> Dict[str, Any]:

        from database.db import (
            STAGE_REPO_SCAN, STAGE_FILE_SELECTION, STAGE_VALIDATION,
            STAGE_DETECTION, STAGE_REPORT,
        )

        ctx = ScanContext.hydrate_from_db(self.db, scan_run_id)

        if stage == STAGE_REPO_SCAN:
            return self.file_selection_agent.scan_repository(ctx)

        if stage == STAGE_FILE_SELECTION:

            with self.db._conn() as conn:
                conn.execute(
                    "DELETE FROM candidate_files WHERE scan_run_id=?",
                    (scan_run_id,),
                )
            return self.file_selection_agent.select_files(
                ctx, max_files=self.max_files,
            )

        if stage == STAGE_VALIDATION:
            with self.db._conn() as conn:
                conn.execute(
                    "UPDATE candidate_files SET status='selected', "
                    "validated_by=NULL, rejection_reason=NULL "
                    "WHERE scan_run_id=? AND status IN ('validated','rejected')",
                    (scan_run_id,),
                )
            return self.validation_agent.validate_files(ctx)

        if stage == STAGE_DETECTION:
            self.db.delete_all_findings(scan_run_id)
            return self.detection_agent.run_detection(ctx)

        if stage == STAGE_REPORT:
            return self.report_agent.generate_report(ctx)

        raise ValueError(f"Unknown stage: {stage!r}")

    def _resolve_empty_evidence(
        self,
        ctx: ScanContext,
        detection_validation: Dict[str, Any] | None,
    ) -> None:

        def _violations(data: Dict[str, Any] | None) -> List[Dict[str, str]]:
            checks = (data or {}).get("checks", {})
            return checks.get("supported_has_evidence", {}).get("violations", []) or []

        attempt = 0
        while attempt < self.max_evidence_retries:
            violations = _violations(detection_validation)
            if not violations:
                return  # contract satisfied — nothing to do

            attempt += 1
            by_practice: Dict[str, List[str]] = {}
            for v in violations:
                by_practice.setdefault(v["practice"], []).append(v["file"])

            self._log(
                f"[Orchestrator] ⚠ {len(violations)} supported finding(s) have "
                f"empty evidence — rerunning detection "
                f"(attempt {attempt}/{self.max_evidence_retries})"
            )

            for practice, files in by_practice.items():
                resp = self.dispatcher.call(
                    "detection_server", "run_detection_on_files",
                    {"scan_run_id": ctx.scan_run_id,
                     "file_paths":  sorted(set(files)),
                     "practices":   [practice]},
                )
                if resp.is_error:
                    self._log(
                        f"[Orchestrator]    rerun failed for "
                        f"'{practice}': {resp.get_text()}"
                    )

            ctx.findings = self.db.get_findings(ctx.scan_run_id)
            detection_validation = self.validation_agent.validate_detection_results(ctx)

        remaining = _violations(detection_validation)
        if remaining:
            practices = sorted({v["practice"] for v in remaining})
            self._log(
                f"[Orchestrator] ✗ {len(remaining)} finding(s) still report "
                f"supported=True with empty evidence after "
                f"{self.max_evidence_retries} rerun(s). This usually means the "
                f"prompt or practice definition is faulty for: {practices}. "
                f"Leaving for human review."
            )

    def _finish(self, ctx: ScanContext) -> Dict[str, Any]:
        self.db.complete_scan_run(ctx.scan_run_id)
        self._mark_complete()

        findings = self.db.get_findings(ctx.scan_run_id)
        from src.output_parser import summarise_findings
        grouped: Dict[str, list] = {}
        for f in findings:
            grouped.setdefault(f["practice"], []).append(f)
        for p in self.detectors:
            grouped.setdefault(p, [])
        results_by_practice = {
            p: {"findings": flist, "summary": summarise_findings(flist)}
            for p, flist in grouped.items()
        }

        return {
            "repo_root":           ctx.repo_root,
            "scan_run_id":         ctx.scan_run_id,
            "candidate_files":     [c["file_path"] for c in
                                    self.db.get_all_candidates(ctx.scan_run_id)],
            "results_by_practice": results_by_practice,
            "stage_status":        self.db.get_stage_status(ctx.scan_run_id),
            "validation_log":      self.db.get_validation_log(ctx.scan_run_id),
            "report_paths":        ctx.report_paths,
            "agent_summary": {
                a["agent_id"]: a["status"] for a in self.db.get_all_agents()
            },
        }
