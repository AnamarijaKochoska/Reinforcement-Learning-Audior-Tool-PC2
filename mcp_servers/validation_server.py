from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from database.db import (
    AuditDatabase,
    STAGE_REPO_SCAN, STAGE_FILE_SELECTION, STAGE_VALIDATION,
    STAGE_DETECTION, STAGE_REPORT,
    STATE_RUNNING, STATE_COMPLETE, STATE_FAILED, STATE_PARTIAL,
)
from mcp_servers.base_server import BaseMCPServer, MCPResponse, register_tool
from src.file_filter import EXTENSION_TO_FILETYPE


ALLOWED_EXTENSIONS = set(EXTENSION_TO_FILETYPE.keys())   # .py, .yaml, .yml, .java
MAX_FILE_SIZE_KB = 500
AGENT_ID = "validation_agent"


class ValidationServer(BaseMCPServer):

    def __init__(self, db: AuditDatabase):
        self.db = db
        super().__init__(
            server_name="validation_server",
            server_description=(
                "Rule-based validation. Handles both pre-detection file "
                "checks and post-detection result sanity checks. "
                "No LLM is involved anywhere in this server."
            ),
        )

    # ── A) Pre-detection file-level checks ─────────────────────────────
    def _validate_single_file(self, file_path: str) -> tuple[bool, str | None]:
        p = Path(file_path)

        if not p.exists():
            return False, "File does not exist on disk"

        if p.suffix.lower() not in ALLOWED_EXTENSIONS:
            return False, (
                f"Extension '{p.suffix}' is not allowed. "
                f"Allowed: {sorted(ALLOWED_EXTENSIONS)}"
            )

        if p.stat().st_size == 0:
            return False, "File is empty"

        size_kb = p.stat().st_size / 1024
        if size_kb > MAX_FILE_SIZE_KB:
            return False, (
                f"File is too large ({size_kb:.1f} KB > {MAX_FILE_SIZE_KB} KB limit)"
            )

        try:
            p.read_text(encoding="utf-8", errors="strict")
        except UnicodeDecodeError:
            try:
                p.read_text(encoding="latin-1")
            except Exception:
                return False, "File is not readable as text (binary file?)"

        return True, None

    @register_tool(
        name="validate_candidates",
        description=(
            "Run file-level validation on every candidate with status='selected' "
            "for the scan run. Updates each file's status to 'validated' or "
            "'rejected'. Updates stage_status for STAGE_VALIDATION."
        ),
        input_schema={
            "type": "object",
            "properties": {"scan_run_id": {"type": "integer"}},
            "required": ["scan_run_id"],
        },
    )
    def validate_candidates(self, scan_run_id: int) -> MCPResponse:
        self.db.set_stage_state(
            scan_run_id, STAGE_VALIDATION, STATE_RUNNING,
            message="running file-level validation",
        )

        selected = self.db.get_candidates_by_status(scan_run_id, "selected")
        if not selected:
            self.db.set_stage_state(
                scan_run_id, STAGE_VALIDATION, STATE_COMPLETE,
                progress={"checked": 0},
                message="no selected candidates to validate",
            )
            return MCPResponse.json_data({
                "status":    "ok",
                "validated": 0,
                "rejected":  0,
                "rejections": [],
                "message":   "No files with status='selected' found.",
            })

        validated_count = 0
        rejected_count = 0
        rejections: List[Dict[str, Any]] = []

        for row in selected:
            is_valid, reason = self._validate_single_file(row["file_path"])
            if is_valid:
                self.db.update_candidate_status(
                    row["id"], "validated", validated_by=AGENT_ID,
                )
                validated_count += 1
            else:
                self.db.update_candidate_status(
                    row["id"], "rejected", validated_by=AGENT_ID,
                    rejection_reason=reason,
                )
                rejected_count += 1
                rejections.append({"file": row["file_path"], "reason": reason})

        final_state = STATE_COMPLETE if rejected_count == 0 else STATE_PARTIAL
        self.db.set_stage_state(
            scan_run_id, STAGE_VALIDATION, final_state,
            progress={"validated": validated_count, "rejected": rejected_count},
            message=f"validated {validated_count}, rejected {rejected_count}",
        )
        self.db.save_checkpoint(
            scan_run_id, STAGE_VALIDATION,
            {
                "validated_files": [
                    r["file_path"] for r in
                    self.db.get_candidates_by_status(scan_run_id, "validated")
                ],
                "rejected_files": rejections,
            },
        )

        return MCPResponse.json_data({
            "status":        "ok",
            "total_checked": len(selected),
            "validated":     validated_count,
            "rejected":      rejected_count,
            "rejections":    rejections,
            "state":         final_state,
        })

    @register_tool(
        name="validate_detection_results",
        description=(
            "Run three sanity checks against the findings in the DB: "
            "(1) every validated file has at least one finding per practice, "
            "(2) every supported=True finding has non-empty evidence, "
            "(3) every supported=False finding has empty evidence. "
            "Writes each check's pass/fail to validation_log. "
            "Does NOT mutate findings — purely reports."
        ),
        input_schema={
            "type": "object",
            "properties": {"scan_run_id": {"type": "integer"}},
            "required": ["scan_run_id"],
        },
    )
    def validate_detection_results(self, scan_run_id: int) -> MCPResponse:
        findings = self.db.get_findings(scan_run_id)
        validated = (
            self.db.get_candidates_by_status(scan_run_id, "validated")
            + self.db.get_candidates_by_status(scan_run_id, "complete")
        )

        check_results: Dict[str, Dict[str, Any]] = {}

        practices_seen = sorted({f["practice"] for f in findings})
        missing_pairs: List[Dict[str, str]] = []
        seen_by_file: Dict[str, set] = {}
        for f in findings:
            seen_by_file.setdefault(f["file_path"], set()).add(f["practice"])
        for v in validated:
            seen = seen_by_file.get(v["file_path"], set())
            for p in practices_seen:
                if p not in seen:
                    missing_pairs.append({"file": v["file_path"], "practice": p})
        c1_passed = len(missing_pairs) == 0
        check_results["all_files_analysed"] = {
            "passed": c1_passed,
            "detail": (
                "All validated files have findings for every practice."
                if c1_passed else
                f"{len(missing_pairs)} (file, practice) pairs missing."
            ),
            "missing_pairs": missing_pairs,
        }
        self.db.log_validation(
            scan_run_id, STAGE_DETECTION, "all_files_analysed",
            c1_passed, check_results["all_files_analysed"]["detail"],
        )

        c2_violations = [
            {"file": f["file_path"], "practice": f["practice"]}
            for f in findings
            if f["supported"] and not f.get("evidence")
        ]
        c2_passed = len(c2_violations) == 0
        check_results["supported_has_evidence"] = {
            "passed": c2_passed,
            "detail": (
                "Every supported=True finding has ≥1 evidence item."
                if c2_passed else
                f"{len(c2_violations)} supported findings have empty evidence."
            ),
            "violations": c2_violations,
        }
        self.db.log_validation(
            scan_run_id, STAGE_DETECTION, "supported_has_evidence",
            c2_passed, check_results["supported_has_evidence"]["detail"],
        )

        c3_violations = [
            {"file": f["file_path"], "practice": f["practice"],
             "evidence_count": len(f.get("evidence", []))}
            for f in findings
            if (not f["supported"]) and f.get("evidence")
        ]
        c3_passed = len(c3_violations) == 0
        check_results["not_supported_has_no_evidence"] = {
            "passed": c3_passed,
            "detail": (
                "Every supported=False finding has empty evidence."
                if c3_passed else
                f"{len(c3_violations)} not-supported findings contain evidence anyway."
            ),
            "violations": c3_violations,
        }
        self.db.log_validation(
            scan_run_id, STAGE_DETECTION, "not_supported_has_no_evidence",
            c3_passed, check_results["not_supported_has_no_evidence"]["detail"],
        )

        overall = all(r["passed"] for r in check_results.values())
        return MCPResponse.json_data({
            "status":           "ok",
            "scan_run_id":      scan_run_id,
            "overall_passed":   overall,
            "checks":           check_results,
            "total_findings":   len(findings),
            "validated_files":  len(validated),
            "practices_seen":   practices_seen,
        })

    @register_tool(
        name="validate_repo_scan_output",
        description=(
            "Two checks after repository_scan: (1) repo_path is a directory, "
            "(2) at least one source file was found. Logs to validation_log."
        ),
        input_schema={
            "type": "object",
            "properties": {"scan_run_id": {"type": "integer"}},
            "required": ["scan_run_id"],
        },
    )
    def validate_repo_scan_output(self, scan_run_id: int) -> MCPResponse:
        run = self.db.get_scan_run(scan_run_id)
        cp = self.db.load_checkpoint(scan_run_id, STAGE_REPO_SCAN)
        files = (cp["payload"].get("all_source_files") if cp else []) or []

        repo_root = run["repo_root"] if run else ""
        c1_passed = bool(repo_root) and os.path.isdir(repo_root)
        c1_detail = (
            f"repo_root '{repo_root}' is a directory."
            if c1_passed else
            f"repo_root '{repo_root}' is not a directory."
        )
        self.db.log_validation(
            scan_run_id, STAGE_REPO_SCAN,
            "repo_path_is_directory", c1_passed, c1_detail,
        )

        c2_passed = len(files) > 0
        c2_detail = (
            f"Found {len(files)} source files."
            if c2_passed else
            "Found 0 source files — downstream stages will have nothing to do."
        )
        self.db.log_validation(
            scan_run_id, STAGE_REPO_SCAN,
            "found_source_files", c2_passed, c2_detail,
        )

        return MCPResponse.json_data({
            "scan_run_id":    scan_run_id,
            "stage":          STAGE_REPO_SCAN,
            "overall_passed": c1_passed and c2_passed,
            "checks": {
                "repo_path_is_directory": {"passed": c1_passed, "detail": c1_detail},
                "found_source_files":     {"passed": c2_passed, "detail": c2_detail},
            },
        })

    @register_tool(
        name="validate_selection_output",
        description=(
            "Two checks after file_selection: (1) at least one file was "
            "selected, (2) every selected file's path actually exists on disk. "
            "Logs to validation_log."
        ),
        input_schema={
            "type": "object",
            "properties": {"scan_run_id": {"type": "integer"}},
            "required": ["scan_run_id"],
        },
    )
    def validate_selection_output(self, scan_run_id: int) -> MCPResponse:
        selected = self.db.get_candidates_by_status(scan_run_id, "selected")

        # Check 1: at least one file selected
        c1_passed = len(selected) > 0
        c1_detail = (
            f"Selected {len(selected)} candidate file(s)."
            if c1_passed else
            "Selected 0 files — keyword filter rejected everything for this repo."
        )
        self.db.log_validation(
            scan_run_id, STAGE_FILE_SELECTION,
            "selected_at_least_one", c1_passed, c1_detail,
        )

        missing = [c["file_path"] for c in selected
                   if not os.path.exists(c["file_path"])]
        c2_passed = len(missing) == 0
        c2_detail = (
            f"All {len(selected)} selected file paths exist on disk."
            if c2_passed else
            f"{len(missing)} selected file(s) no longer exist on disk: "
            f"{missing[:3]}{'…' if len(missing) > 3 else ''}"
        )
        self.db.log_validation(
            scan_run_id, STAGE_FILE_SELECTION,
            "selection_paths_exist", c2_passed, c2_detail,
        )

        return MCPResponse.json_data({
            "scan_run_id":    scan_run_id,
            "stage":          STAGE_FILE_SELECTION,
            "overall_passed": c1_passed and c2_passed,
            "checks": {
                "selected_at_least_one": {"passed": c1_passed, "detail": c1_detail},
                "selection_paths_exist": {"passed": c2_passed, "detail": c2_detail},
            },
        })

    @register_tool(
        name="validate_validation_output",
        description=(
            "Two checks after the file-validation stage: (1) at least one "
            "file passed validation, (2) not every file was rejected. "
            "Logs to validation_log."
        ),
        input_schema={
            "type": "object",
            "properties": {"scan_run_id": {"type": "integer"}},
            "required": ["scan_run_id"],
        },
    )
    def validate_validation_output(self, scan_run_id: int) -> MCPResponse:
        validated = (
            self.db.get_candidates_by_status(scan_run_id, "validated")
            + self.db.get_candidates_by_status(scan_run_id, "complete")
        )
        rejected = self.db.get_candidates_by_status(scan_run_id, "rejected")
        total = len(validated) + len(rejected)

        c1_passed = len(validated) > 0
        c1_detail = (
            f"{len(validated)} file(s) passed file-level validation."
            if c1_passed else
            "0 files passed validation — detection will have nothing to do."
        )
        self.db.log_validation(
            scan_run_id, STAGE_VALIDATION,
            "at_least_one_validated", c1_passed, c1_detail,
        )

        rejection_rate = (len(rejected) / total) if total else 0.0
        c2_passed = rejection_rate < 1.0
        c2_detail = (
            f"Rejection rate: {len(rejected)}/{total} "
            f"({rejection_rate * 100:.0f}%)."
        )
        self.db.log_validation(
            scan_run_id, STAGE_VALIDATION,
            "not_all_rejected", c2_passed, c2_detail,
        )

        return MCPResponse.json_data({
            "scan_run_id":    scan_run_id,
            "stage":          STAGE_VALIDATION,
            "overall_passed": c1_passed and c2_passed,
            "checks": {
                "at_least_one_validated": {"passed": c1_passed, "detail": c1_detail},
                "not_all_rejected":       {"passed": c2_passed, "detail": c2_detail},
            },
        })

    @register_tool(
        name="validate_report_output",
        description=(
            "Three checks after report_generation: (1) report paths were "
            "recorded, (2) all recorded files exist on disk, (3) all files "
            "are non-empty. Logs to validation_log."
        ),
        input_schema={
            "type": "object",
            "properties": {"scan_run_id": {"type": "integer"}},
            "required": ["scan_run_id"],
        },
    )
    def validate_report_output(self, scan_run_id: int) -> MCPResponse:
        cp = self.db.load_checkpoint(scan_run_id, STAGE_REPORT)
        paths = (cp["payload"].get("report_paths") if cp else {}) or {}

        # Check 1: paths recorded
        c1_passed = len(paths) > 0
        c1_detail = (
            f"Recorded {len(paths)} report path(s): {list(paths.keys())}."
            if c1_passed else
            "No report paths recorded — report stage didn't produce a checkpoint."
        )
        self.db.log_validation(
            scan_run_id, STAGE_REPORT,
            "report_paths_recorded", c1_passed, c1_detail,
        )

        missing = [k for k, v in paths.items() if not os.path.exists(v)]
        c2_passed = len(missing) == 0 if paths else False
        c2_detail = (
            "All recorded report files exist on disk."
            if c2_passed else
            f"Missing report file(s): {missing}"
        )
        self.db.log_validation(
            scan_run_id, STAGE_REPORT,
            "report_files_exist", c2_passed, c2_detail,
        )


        empty = [k for k, v in paths.items()
                 if os.path.exists(v) and os.path.getsize(v) == 0]
        c3_passed = len(empty) == 0 if paths else False
        c3_detail = (
            "All report files are non-empty."
            if c3_passed else
            f"Empty report file(s): {empty}"
        )
        self.db.log_validation(
            scan_run_id, STAGE_REPORT,
            "report_files_non_empty", c3_passed, c3_detail,
        )

        return MCPResponse.json_data({
            "scan_run_id":    scan_run_id,
            "stage":          STAGE_REPORT,
            "overall_passed": c1_passed and c2_passed and c3_passed,
            "checks": {
                "report_paths_recorded":  {"passed": c1_passed, "detail": c1_detail},
                "report_files_exist":     {"passed": c2_passed, "detail": c2_detail},
                "report_files_non_empty": {"passed": c3_passed, "detail": c3_detail},
            },
        })

    @register_tool(
        name="get_validation_report",
        description=(
            "Return a consolidated validation summary for a scan run — "
            "both file-level validation results and the logged post-stage "
            "result checks."
        ),
        input_schema={
            "type": "object",
            "properties": {"scan_run_id": {"type": "integer"}},
            "required": ["scan_run_id"],
        },
    )
    def get_validation_report(self, scan_run_id: int) -> MCPResponse:
        validated = self.db.get_candidates_by_status(scan_run_id, "validated")
        completed = self.db.get_candidates_by_status(scan_run_id, "complete")
        rejected  = self.db.get_candidates_by_status(scan_run_id, "rejected")
        log       = self.db.get_validation_log(scan_run_id)

        return MCPResponse.json_data({
            "scan_run_id":     scan_run_id,
            "validated_files": [r["file_path"] for r in validated + completed],
            "rejected_files": [
                {"file": r["file_path"], "reason": r.get("rejection_reason")}
                for r in rejected
            ],
            "validated_count": len(validated) + len(completed),
            "rejected_count":  len(rejected),
            "validation_log":  log,
        })
