from __future__ import annotations
from typing import Any, Dict

from agents.base_agent import BaseAgent
from memory.short_term import ScanContext


class ValidationAgent(BaseAgent):
    agent_id   = "validation_agent"
    agent_type = "validation"

    def validate_files(self, ctx: ScanContext) -> Dict[str, Any]:
        self._mark_running(f"validate_files:scan={ctx.scan_run_id}")
        resp = self._call(
            "validation_server", "validate_candidates",
            {"scan_run_id": ctx.scan_run_id},
        )
        if resp.is_error:
            self._mark_error(resp.get_text())
            raise RuntimeError(f"file validation failed: {resp.get_text()}")
        data = resp.get_data()
        ctx.validated_files = self.db.get_candidates_by_status(
            ctx.scan_run_id, "validated",
        )
        ctx.rejected_files = self.db.get_candidates_by_status(
            ctx.scan_run_id, "rejected",
        )
        ctx.note(
            f"validate_files: {data['validated']} ok, {data['rejected']} rejected"
        )
        self._log(
            f"[ValidationAgent] file validation: "
            f"{data['validated']} validated, {data['rejected']} rejected"
        )
        for r in data.get("rejections", []):
            self._log(f"  ✗ {r['file']} — {r['reason']}")
        self._mark_complete()
        return data

    def _run_stage_validation(
        self,
        ctx: ScanContext,
        tool_name: str,
        stage_label: str,
    ) -> Dict[str, Any]:
        """
        Shared implementation: call the named tool, log a per-check
        summary line, and return its result data. Failures are warnings,
        never raised — the pipeline keeps moving.
        """
        self._mark_running(f"{tool_name}:scan={ctx.scan_run_id}")
        resp = self._call(
            "validation_server", tool_name,
            {"scan_run_id": ctx.scan_run_id},
        )
        if resp.is_error:
            # Even tool errors don't halt — we log and continue.
            self._log(
                f"[ValidationAgent] {stage_label} validation tool error: "
                f"{resp.get_text()}"
            )
            self._mark_complete()
            return {"overall_passed": False, "checks": {},
                    "error": resp.get_text()}

        data = resp.get_data()
        passed = data.get("overall_passed", False)
        self._log(
            f"[ValidationAgent] {stage_label} checks: "
            f"{'all passed' if passed else 'WARNINGS PRESENT'}"
        )
        for name, result in data.get("checks", {}).items():
            icon = "✓" if result.get("passed") else "✗"
            self._log(f"  {icon}  {name:<34} {result.get('detail', '')}")
        self._mark_complete()
        return data

    def validate_repo_scan_output(self, ctx: ScanContext) -> Dict[str, Any]:
        return self._run_stage_validation(
            ctx, "validate_repo_scan_output", "repo_scan",
        )

    def validate_selection_output(self, ctx: ScanContext) -> Dict[str, Any]:
        return self._run_stage_validation(
            ctx, "validate_selection_output", "selection",
        )

    def validate_validation_output(self, ctx: ScanContext) -> Dict[str, Any]:
        return self._run_stage_validation(
            ctx, "validate_validation_output", "file_validation",
        )

    def validate_detection_results(self, ctx: ScanContext) -> Dict[str, Any]:
        return self._run_stage_validation(
            ctx, "validate_detection_results", "detection",
        )

    def validate_report_output(self, ctx: ScanContext) -> Dict[str, Any]:
        return self._run_stage_validation(
            ctx, "validate_report_output", "report",
        )

    def validate_results(self, ctx: ScanContext) -> Dict[str, Any]:
        return self.validate_detection_results(ctx)
