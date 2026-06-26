from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from database.db import (
    AuditDatabase,
    STAGE_DETECTION,
    STATE_RUNNING, STATE_COMPLETE, STATE_PARTIAL, STATE_FAILED,
)
from mcp_servers.base_server import BaseMCPServer, MCPResponse, register_tool
from src.llm import OllamaClient
from src.file_filter import extract_relevant_sections, classify_file_type
from src.output_parser import parse_llm_response, summarise_findings

_FENCE_LABEL = {
    "python": "python",
    "java":   "java",
    "yaml":   "yaml",
    "unknown": "",
}


def _add_line_numbers(text: str) -> str:
    return "\n".join(
        f"{i + 1:04d}: {line}" for i, line in enumerate(text.splitlines())
    )


def _build_file_context(file_path: str) -> tuple[str, bool]:
    text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    text, was_extracted = extract_relevant_sections(text)
    numbered = _add_line_numbers(text)

    ftype = classify_file_type(file_path)
    fence = _FENCE_LABEL.get(ftype, "")
    lang_hint = f" ({ftype})" if ftype != "unknown" else ""

    context = (
        f"\n\nTARGET FILE{lang_hint}:\n"
        f"```{fence}\n# FILE: {file_path}\n{numbered}\n```\n"
    )
    return context, was_extracted


class DetectionServer(BaseMCPServer):

    def __init__(
        self,
        db: AuditDatabase,
        llm_client: OllamaClient,
        detectors: Dict[str, Any],
    ):
        self.db = db
        self.llm = llm_client
        self.detectors = detectors   # {practice_name: conversation_fn}
        super().__init__(
            server_name="detection_server",
            server_description=(
                "Runs LLM-based RL data-collection practice detection on "
                "validated candidate files. Persists structured evidence "
                "(line numbers, snippets, explanations) plus LLM-provided "
                "assets/notes. Does NOT generate reports — see "
                "report_generator_server."
            ),
        )
    def _analyse_file(
        self, practice_name: str, conversation_fn, file_path: str,
    ) -> Dict[str, Any]:
        try:
            target_context, was_extracted = _build_file_context(file_path)
        except OSError as exc:
            return {
                "file": file_path, "supported": False, "evidence": [],
                "assets": {}, "was_extracted": False, "raw_response": "",
                "parse_warning": f"Could not read file: {exc}",
            }

        conversation = conversation_fn(target_context)
        try:
            raw_response = self.llm.chat(conversation)
        except Exception as exc:
            return {
                "file": file_path, "supported": False, "evidence": [],
                "assets": {}, "was_extracted": was_extracted, "raw_response": "",
                "parse_warning": f"LLM call failed: {exc}",
            }

        finding = parse_llm_response(file_path, raw_response)
        finding["was_extracted"] = was_extracted
        return finding

    @register_tool(
        name="run_detection",
        description=(
            "Run every registered practice detector against every validated "
            "file for the scan run. Writes one finding per (file, practice) "
            "pair to the findings table. Returns a per-practice summary. "
            "Updates stage_status for STAGE_DETECTION."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "scan_run_id": {"type": "integer"},
                "practices":   {"type": "array", "items": {"type": "string"},
                                "description": (
                                    "Optional subset of practice names to run. "
                                    "Defaults to all registered detectors."
                                )},
            },
            "required": ["scan_run_id"],
        },
    )
    def run_detection(
        self,
        scan_run_id: int,
        practices: list | None = None,
    ) -> MCPResponse:
        self.db.set_stage_state(
            scan_run_id, STAGE_DETECTION, STATE_RUNNING,
            message="running detectors",
        )
        validated = (
            self.db.get_candidates_by_status(scan_run_id, "validated")
            + self.db.get_candidates_by_status(scan_run_id, "complete")
        )
        if not validated:
            self.db.set_stage_state(
                scan_run_id, STAGE_DETECTION, STATE_COMPLETE,
                progress={"processed": 0, "total": 0},
                message="no validated files — nothing to analyse",
            )
            return MCPResponse.json_data({
                "status":  "ok",
                "message": "No validated files to process.",
                "results_by_practice": {},
            })

        to_run = (
            {k: self.detectors[k] for k in practices if k in self.detectors}
            if practices else dict(self.detectors)
        )
        if not to_run:
            self.db.set_stage_state(
                scan_run_id, STAGE_DETECTION, STATE_FAILED,
                message="no matching practices requested",
            )
            return MCPResponse.error(
                f"None of the requested practices are registered. "
                f"Available: {list(self.detectors.keys())}"
            )

        total_tasks = len(validated) * len(to_run)
        processed = 0
        failed = 0
        results_by_practice: Dict[str, Any] = {}

        for practice_name, conv_fn in to_run.items():
            print(f"  [DetectionServer] Practice: {practice_name}")
            findings_list: List[Dict[str, Any]] = []

            for row in validated:
                fp = row["file_path"]
                print(f"    Analysing: {Path(fp).name}")
                finding = self._analyse_file(practice_name, conv_fn, fp)

                self.db.insert_finding(
                    scan_run_id=scan_run_id,
                    file_path=fp,
                    practice=practice_name,
                    supported=finding["supported"],
                    evidence=finding.get("evidence", []),
                    assets=finding.get("assets", {}),
                    raw_response=finding.get("raw_response", ""),
                    parse_warning=finding.get("parse_warning"),
                    was_extracted=finding.get("was_extracted", False),
                )

                if finding.get("parse_warning"):
                    failed += 1
                processed += 1

                self.db.set_stage_state(
                    scan_run_id, STAGE_DETECTION, STATE_PARTIAL,
                    progress={"processed": processed, "total": total_tasks,
                              "warnings": failed},
                    message=f"{processed}/{total_tasks} file×practice pairs",
                )

                status_icon = "✓" if finding["supported"] else "✗"
                warn = " ⚠" if finding.get("parse_warning") else ""
                print(f"      → {status_icon} "
                      f"{'supported' if finding['supported'] else 'not supported'}{warn}")
                findings_list.append(finding)

            results_by_practice[practice_name] = {
                "findings": findings_list,
                "summary":  summarise_findings(findings_list),
            }
        for row in validated:
            self.db.update_candidate_status(row["id"], "complete")

        final_state = STATE_COMPLETE if failed == 0 else STATE_PARTIAL
        final_msg = (
            f"analysed {processed} pairs"
            + (f" ({failed} with parse warnings)" if failed else "")
        )
        self.db.set_stage_state(
            scan_run_id, STAGE_DETECTION, final_state,
            progress={"processed": processed, "total": total_tasks,
                      "warnings": failed},
            message=final_msg,
        )
        self.db.save_checkpoint(
            scan_run_id, STAGE_DETECTION,
            {
                "files_analyzed": [r["file_path"] for r in validated],
                "practices_run":  list(to_run.keys()),
                "warnings":       failed,
            },
        )

        return MCPResponse.json_data({
            "status":              "ok",
            "scan_run_id":         scan_run_id,
            "state":               final_state,
            "total_pairs":         total_tasks,
            "warnings":            failed,
            "results_by_practice": results_by_practice,
        })

    @register_tool(
        name="run_detection_on_files",
        description=(
            "Run detection against a specific list of file paths (which must "
            "already exist as candidates in the DB). Useful for re-running "
            "detection on a subset without re-doing the whole scan."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "scan_run_id": {"type": "integer"},
                "file_paths":  {"type": "array", "items": {"type": "string"}},
                "practices":   {"type": "array", "items": {"type": "string"}},
            },
            "required": ["scan_run_id", "file_paths"],
        },
    )
    def run_detection_on_files(
        self,
        scan_run_id: int,
        file_paths: list,
        practices: list | None = None,
    ) -> MCPResponse:
        all_cands = self.db.get_all_candidates(scan_run_id)
        wanted = set(file_paths)
        subset = [c for c in all_cands if c["file_path"] in wanted]
        if not subset:
            return MCPResponse.error(
                "None of the given file_paths are candidates for this scan run."
            )

        to_run = (
            {k: self.detectors[k] for k in practices if k in self.detectors}
            if practices else dict(self.detectors)
        )

        replaced = 0
        for row in subset:
            for practice_name in to_run:
                replaced += self.db.delete_findings_for_file(
                    scan_run_id, row["file_path"], practice_name,
                )

        results: Dict[str, Any] = {}
        for practice_name, conv_fn in to_run.items():
            findings_list = []
            for row in subset:
                finding = self._analyse_file(practice_name, conv_fn, row["file_path"])
                self.db.insert_finding(
                    scan_run_id=scan_run_id,
                    file_path=row["file_path"],
                    practice=practice_name,
                    supported=finding["supported"],
                    evidence=finding.get("evidence", []),
                    assets=finding.get("assets", {}),
                    raw_response=finding.get("raw_response", ""),
                    parse_warning=finding.get("parse_warning"),
                    was_extracted=finding.get("was_extracted", False),
                )
                findings_list.append(finding)
            results[practice_name] = {
                "findings": findings_list,
                "summary":  summarise_findings(findings_list),
            }

        return MCPResponse.json_data({
            "status":              "ok",
            "scan_run_id":         scan_run_id,
            "files_processed":     [c["file_path"] for c in subset],
            "rows_replaced":       replaced,
            "results_by_practice": results,
        })

    @register_tool(
        name="get_findings",
        description="Return all stored findings for a scan run, grouped by practice.",
        input_schema={
            "type": "object",
            "properties": {"scan_run_id": {"type": "integer"}},
            "required": ["scan_run_id"],
        },
    )
    def get_findings(self, scan_run_id: int) -> MCPResponse:
        findings = self.db.get_findings(scan_run_id)

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for f in findings:
            grouped.setdefault(f["practice"], []).append(f)

        structured = {
            p: {"findings": flist, "summary": summarise_findings(flist)}
            for p, flist in grouped.items()
        }
        return MCPResponse.json_data({
            "scan_run_id":         scan_run_id,
            "total_findings":      len(findings),
            "results_by_practice": structured,
        })

    @register_tool(
        name="get_detection_status",
        description="Return the detection-stage state for a scan run.",
        input_schema={
            "type": "object",
            "properties": {"scan_run_id": {"type": "integer"}},
            "required": ["scan_run_id"],
        },
    )
    def get_detection_status(self, scan_run_id: int) -> MCPResponse:
        row = self.db.get_stage_status(scan_run_id, STAGE_DETECTION)
        if not row:
            return MCPResponse.error(f"No scan run with id={scan_run_id}")
        return MCPResponse.json_data(row)
