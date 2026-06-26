from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from database.db import (
    AuditDatabase,
    STAGE_REPO_SCAN,
    STATE_RUNNING, STATE_COMPLETE, STATE_FAILED,
)
from mcp_servers.base_server import BaseMCPServer, MCPResponse, register_tool
from src.file_filter import (
    collect_source_files,
    classify_file_type,
    EXTENSION_TO_FILETYPE,
)


class RepositoryScannerServer(BaseMCPServer):

    def __init__(self, db: AuditDatabase):
        self.db = db
        super().__init__(
            server_name="repository_scanner_server",
            server_description=(
                "Walks a directory tree and produces a per-language inventory "
                "of source files. Does NOT apply keyword filtering — that is "
                "the next stage's job."
            ),
        )

    @register_tool(
        name="scan_repository",
        description=(
            "Recursively walk `repo_root`, collect all supported source "
            "files (.py, .yaml, .yml, .java by default), and record a "
            "checkpoint with per-language counts. Updates stage_status for "
            "STAGE_REPO_SCAN to 'running' → 'complete' (or 'failed')."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "repo_root":   {"type": "string",
                                "description": "Absolute path to the repository"},
                "scan_run_id": {"type": "integer",
                                "description": "ID of the current scan run"},
                "extensions":  {"type": "array", "items": {"type": "string"},
                                "description": (
                                    "Optional override list of extensions "
                                    "(e.g. [\".py\", \".java\"]). Defaults to "
                                    "all known source extensions."
                                )},
            },
            "required": ["repo_root", "scan_run_id"],
        },
    )
    def scan_repository(
        self,
        repo_root: str,
        scan_run_id: int,
        extensions: list | None = None,
    ) -> MCPResponse:
        import os

        self.db.set_stage_state(
            scan_run_id, STAGE_REPO_SCAN, STATE_RUNNING,
            message=f"scanning {repo_root}",
        )

        if not os.path.isdir(repo_root):
            self.db.set_stage_state(
                scan_run_id, STAGE_REPO_SCAN, STATE_FAILED,
                message=f"not a directory: {repo_root}",
            )
            return MCPResponse.error(f"Not a directory: {repo_root}")

        ext_set = set(extensions) if extensions else set(EXTENSION_TO_FILETYPE.keys())
        all_files = collect_source_files(repo_root, ext_set)

        by_type: dict[str, int] = {}
        for p in all_files:
            by_type[classify_file_type(p)] = by_type.get(classify_file_type(p), 0) + 1

        summary = {
            "repo_root":     repo_root,
            "total_files":   len(all_files),
            "by_file_type":  by_type,
            "extensions":    sorted(ext_set),
        }

        self.db.save_checkpoint(
            scan_run_id, STAGE_REPO_SCAN,
            {"all_source_files": all_files, "repo_scan_summary": summary},
        )
        self.db.set_stage_state(
            scan_run_id, STAGE_REPO_SCAN, STATE_COMPLETE,
            progress={"processed": len(all_files), "total": len(all_files)},
            message=f"found {len(all_files)} source files",
        )

        return MCPResponse.json_data({
            "status":      "ok",
            "scan_run_id": scan_run_id,
            "summary":     summary,
            "files":       all_files,
        })

    @register_tool(
        name="get_repo_scan_summary",
        description=(
            "Return the repo-scan checkpoint for a scan run. Returns an "
            "error if the repo_scan stage hasn't run yet."
        ),
        input_schema={
            "type": "object",
            "properties": {"scan_run_id": {"type": "integer"}},
            "required": ["scan_run_id"],
        },
    )
    def get_repo_scan_summary(self, scan_run_id: int) -> MCPResponse:
        cp = self.db.load_checkpoint(scan_run_id, STAGE_REPO_SCAN)
        if not cp:
            return MCPResponse.error(
                f"No repo-scan checkpoint for scan_run_id={scan_run_id}. "
                "Run 'scan_repository' first."
            )
        return MCPResponse.json_data({
            "status":      "ok",
            "scan_run_id": scan_run_id,
            "checkpoint":  cp,
        })
