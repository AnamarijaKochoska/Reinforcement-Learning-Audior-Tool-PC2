from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from database.db import (
    AuditDatabase,
    STAGE_FILE_SELECTION, STAGE_REPO_SCAN,
    STATE_RUNNING, STATE_COMPLETE, STATE_PARTIAL, STATE_FAILED,
)
from mcp_servers.base_server import BaseMCPServer, MCPResponse, register_tool
from src.file_filter import (
    is_skippable_path,
    passes_keyword_filter,
    keyword_score,
    classify_file_type,
    collect_source_files,
)


class FileSelectionServer(BaseMCPServer):

    def __init__(self, db: AuditDatabase):
        self.db = db
        super().__init__(
            server_name="file_selection_server",
            server_description=(
                "Ranks and selects candidate files from the repo-scan "
                "inventory using RL-keyword pre-filtering. Consumes the "
                "repository_scan checkpoint if available."
            ),
        )

    def _get_source_inventory(
        self, scan_run_id: int, repo_root: str | None,
    ) -> list[str]:
        cp = self.db.load_checkpoint(scan_run_id, STAGE_REPO_SCAN)
        if cp and cp["payload"].get("all_source_files"):
            return cp["payload"]["all_source_files"]
        if repo_root:
            return collect_source_files(repo_root)
        return []

    @register_tool(
        name="select_files",
        description=(
            "Apply keyword pre-filtering to the repo-scan inventory, rank by "
            "relevance, cap at `max_files`, and persist candidates with "
            "status='selected'. Updates stage_status for STAGE_FILE_SELECTION "
            "to 'partial' if the cap truncated the list, otherwise 'complete'."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "scan_run_id": {"type": "integer"},
                "max_files":   {"type": "integer",
                                "description": (
                                    "Cap on candidates. 0 / negative / omitted "
                                    "means NO LIMIT (select every file that "
                                    "passes the keyword filter)."
                                )},
                "repo_root":   {"type": "string",
                                "description": (
                                    "Optional. Only used as a fallback if no "
                                    "repo_scan checkpoint exists yet."
                                )},
            },
            "required": ["scan_run_id"],
        },
    )
    def select_files(
        self,
        scan_run_id: int,
        max_files: int | None = None,
        repo_root: str | None = None,
    ) -> MCPResponse:
        self.db.set_stage_state(
            scan_run_id, STAGE_FILE_SELECTION, STATE_RUNNING,
            message="filtering + ranking",
        )

        all_files = self._get_source_inventory(scan_run_id, repo_root)
        if not all_files:
            self.db.set_stage_state(
                scan_run_id, STAGE_FILE_SELECTION, STATE_FAILED,
                message="no repo_scan checkpoint and no repo_root given",
            )
            return MCPResponse.error(
                "No source inventory available. Run 'scan_repository' first "
                "or pass `repo_root` for a direct fallback scan."
            )

        candidates = [
            p for p in all_files
            if not is_skippable_path(p) and passes_keyword_filter(p)
        ]
        candidates.sort(key=keyword_score, reverse=True)
        no_cap = (max_files is None) or (max_files <= 0)
        truncated = (not no_cap) and len(candidates) > max_files
        if not no_cap:
            candidates = candidates[:max_files]

        inserted = 0
        for path in candidates:
            self.db.insert_candidate(
                scan_run_id=scan_run_id,
                file_path=path,
                file_type=classify_file_type(path),
                keyword_score=keyword_score(path),
            )
            inserted += 1

        self.db.update_scan_run_candidates(scan_run_id, inserted)

        summary = {
            "total_source_files":    len(all_files),
            "candidates_after_filter": len(candidates),
            "candidates_stored":     inserted,
            "truncated_by_max_files": truncated,
            "max_files":             ("unlimited" if no_cap else max_files),
        }
        state = STATE_PARTIAL if truncated else STATE_COMPLETE
        self.db.set_stage_state(
            scan_run_id, STAGE_FILE_SELECTION, state,
            progress={"selected": inserted, "discovered": len(all_files)},
            message=(f"selected {inserted}/{len(all_files)} "
                     f"({'capped' if truncated else 'all that qualified'})"),
        )
        self.db.save_checkpoint(
            scan_run_id, STAGE_FILE_SELECTION,
            {"selected_files": candidates, "selection_summary": summary},
        )

        return MCPResponse.json_data({
            "status":      "ok",
            "scan_run_id": scan_run_id,
            "state":       state,
            "summary":     summary,
            "selected":    candidates,
        })

    @register_tool(
        name="list_candidates",
        description="Return all candidate files for a scan run, optionally filtered by status.",
        input_schema={
            "type": "object",
            "properties": {
                "scan_run_id": {"type": "integer"},
                "status":      {"type": "string",
                                "description": (
                                    "Filter by status "
                                    "(selected|validated|rejected|complete). "
                                    "Omit for all."
                                )},
            },
            "required": ["scan_run_id"],
        },
    )
    def list_candidates(
        self, scan_run_id: int, status: str | None = None,
    ) -> MCPResponse:
        if status:
            rows = self.db.get_candidates_by_status(scan_run_id, status)
        else:
            rows = self.db.get_all_candidates(scan_run_id)
        return MCPResponse.json_data({
            "scan_run_id": scan_run_id,
            "count":       len(rows),
            "candidates":  rows,
        })

    @register_tool(
        name="get_selection_status",
        description=(
            "Return the current file-selection state for a scan run — "
            "one of not_started / running / partial / complete / failed — "
            "plus progress counters."
        ),
        input_schema={
            "type": "object",
            "properties": {"scan_run_id": {"type": "integer"}},
            "required": ["scan_run_id"],
        },
    )
    def get_selection_status(self, scan_run_id: int) -> MCPResponse:
        row = self.db.get_stage_status(scan_run_id, STAGE_FILE_SELECTION)
        if not row:
            return MCPResponse.error(f"No scan run with id={scan_run_id}")
        return MCPResponse.json_data(row)
