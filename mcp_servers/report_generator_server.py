from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Dict

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from database.db import (
    AuditDatabase,
    STAGE_REPORT,
    STATE_RUNNING, STATE_COMPLETE, STATE_FAILED,
)
from mcp_servers.base_server import BaseMCPServer, MCPResponse, register_tool
from src.output_parser import summarise_findings
from src.report_generator import ReportGenerator


class ReportGeneratorServer(BaseMCPServer):

    def __init__(self, db: AuditDatabase, reports_dir: str = "reports"):
        self.db = db
        self.reports_dir = reports_dir
        super().__init__(
            server_name="report_generator_server",
            server_description=(
                "Builds JSON and HTML reports from findings stored by the "
                "detection server. Runs fully offline (no LLM needed) and "
                "can be invoked standalone for any existing scan_run_id."
            ),
        )

    def _assemble_scan_result(self, scan_run_id: int) -> Dict[str, Any]:
        run = self.db.get_scan_run(scan_run_id)
        if run is None:
            raise ValueError(f"No scan run with id={scan_run_id}")

        findings = self.db.get_findings(scan_run_id)
        grouped: Dict[str, list] = {}
        for f in findings:
            grouped.setdefault(f["practice"], []).append(f)

        results_by_practice = {
            p: {"findings": flist, "summary": summarise_findings(flist)}
            for p, flist in grouped.items()
        }

        all_candidates = self.db.get_all_candidates(scan_run_id)
        stage_status = self.db.get_stage_status(scan_run_id)

        return {
            "repo_root":           run["repo_root"],
            "scan_run_id":         scan_run_id,
            "model":               run["model"],
            "candidate_files":     [c["file_path"] for c in all_candidates],
            "results_by_practice": results_by_practice,
            "stage_status":        stage_status,
            "agent_summary": {
                a["agent_id"]: a["status"] for a in self.db.get_all_agents()
            },
        }

    @register_tool(
        name="generate_report",
        description=(
            "Assemble findings for a scan run into JSON + HTML reports in "
            "the configured reports directory. Returns the output paths."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "scan_run_id": {"type": "integer"},
                "format":      {"type": "string",
                                "description": (
                                    "One of 'json', 'html', or 'all'. "
                                    "Default: 'all'."
                                )},
                "stem":        {"type": "string",
                                "description": "Filename stem. Default: 'scan'."},
            },
            "required": ["scan_run_id"],
        },
    )
    def generate_report(
        self,
        scan_run_id: int,
        format: str = "all",
        stem: str = "scan",
    ) -> MCPResponse:
        self.db.set_stage_state(
            scan_run_id, STAGE_REPORT, STATE_RUNNING,
            message=f"building {format} report(s)",
        )
        try:
            scan_result = self._assemble_scan_result(scan_run_id)
        except ValueError as exc:
            self.db.set_stage_state(
                scan_run_id, STAGE_REPORT, STATE_FAILED, message=str(exc),
            )
            return MCPResponse.error(str(exc))

        gen = ReportGenerator(self.reports_dir)
        fmt = (format or "all").lower()
        paths: Dict[str, str] = {}
        if fmt == "json":
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            paths["json"] = str(gen.save_json(scan_result, f"{stem}_{ts}.json"))
        elif fmt == "html":
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            paths["html"] = str(gen.save_html(scan_result, f"{stem}_{ts}.html"))
        else:
            paths = {k: str(v) for k, v in gen.save_all(scan_result, stem).items()}

        self.db.save_checkpoint(
            scan_run_id, STAGE_REPORT, {"report_paths": paths},
        )
        self.db.set_stage_state(
            scan_run_id, STAGE_REPORT, STATE_COMPLETE,
            progress={"formats": list(paths.keys())},
            message=f"wrote {', '.join(paths.keys())}",
        )

        return MCPResponse.json_data({
            "status":      "ok",
            "scan_run_id": scan_run_id,
            "report_paths": paths,
        })

    @register_tool(
        name="get_report_paths",
        description="Return the most recently written report paths for a scan run.",
        input_schema={
            "type": "object",
            "properties": {"scan_run_id": {"type": "integer"}},
            "required": ["scan_run_id"],
        },
    )
    def get_report_paths(self, scan_run_id: int) -> MCPResponse:
        cp = self.db.load_checkpoint(scan_run_id, STAGE_REPORT)
        if not cp:
            return MCPResponse.error(
                f"No report has been generated for scan_run_id={scan_run_id}."
            )
        return MCPResponse.json_data({
            "scan_run_id":  scan_run_id,
            "report_paths": cp["payload"].get("report_paths", {}),
            "generated_at": cp["created_at"],
        })

    @register_tool(
        name="get_report_status",
        description="Return the report-stage state for a scan run.",
        input_schema={
            "type": "object",
            "properties": {"scan_run_id": {"type": "integer"}},
            "required": ["scan_run_id"],
        },
    )
    def get_report_status(self, scan_run_id: int) -> MCPResponse:
        row = self.db.get_stage_status(scan_run_id, STAGE_REPORT)
        if not row:
            return MCPResponse.error(f"No scan run with id={scan_run_id}")
        return MCPResponse.json_data(row)
