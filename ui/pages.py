from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import streamlit as st

from database.db import (
    AuditDatabase,
    STAGE_REPO_SCAN, STAGE_FILE_SELECTION, STAGE_VALIDATION,
    STAGE_DETECTION, STAGE_REPORT,
)
from src.output_parser import summarise_findings
from ui.components import (
    section_header,
    metric_row,
    stage_status_table,
    state_badge,
    practice_summary_card,
    evidence_card,
    validation_check_row,
    pipeline_stage_card,
    overwrite_notice,
)

_STAGE_ORDER = [
    (STAGE_REPO_SCAN,      "1. Repository scan"),
    (STAGE_FILE_SELECTION, "2. File selection"),
    (STAGE_VALIDATION,     "3. File validation"),
    (STAGE_DETECTION,      "4. Detection (LLM)"),
    (STAGE_REPORT,         "5. Report generation"),
]

def render_overview(
    db: AuditDatabase,
    scan_run_id: int,
    get_orchestrator: Optional[Callable] = None,
) -> None:
    run = db.get_scan_run(scan_run_id)
    if not run:
        st.warning("No data for this scan run.")
        return

    section_header("Run details")
    st.markdown(
        f"""
        <div style='font-family:monospace;font-size:0.85rem;color:#aaa;
                    background:#16213e;padding:0.8rem 1rem;border-radius:4px;'>
        <b style='color:#eee;'>Scan ID:</b> {scan_run_id}<br/>
        <b style='color:#eee;'>Repo:</b> {run['repo_root']}<br/>
        <b style='color:#eee;'>Model:</b> {run.get('model') or '—'}<br/>
        <b style='color:#eee;'>Started:</b> {run.get('started_at') or '—'}<br/>
        <b style='color:#eee;'>Completed:</b> {run.get('completed_at') or '—'}<br/>
        <b style='color:#eee;'>Status:</b> {run.get('status') or '—'}
        </div>
        """,
        unsafe_allow_html=True,
    )

    candidates = db.get_all_candidates(scan_run_id)
    findings   = db.get_findings(scan_run_id)
    validated  = [c for c in candidates if c["status"] in ("validated", "complete")]
    supported  = [f for f in findings if f["supported"]]

    section_header("Summary")
    metric_row({
        "Files discovered": run.get("total_candidates") or len(candidates),
        "Files validated":  len(validated),
        "Total findings":   len(findings),
        "Supported":        len(supported),
    })

    val_log = db.get_validation_log(scan_run_id)
    if val_log:
        passed = sum(1 for r in val_log if r["passed"])
        total  = len(val_log)
        all_ok = passed == total
        color  = "#2ecc71" if all_ok else "#f1c40f"
        label  = "ALL PASSED" if all_ok else "WARNINGS"
        st.markdown(
            f"""
            <div style='background:#16213e;border-left:5px solid {color};
                        padding:0.7rem 1rem;border-radius:4px;margin:0.5rem 0 1rem;'>
              <div style='display:flex;justify-content:space-between;align-items:center;'>
                <div style='color:#eee;font-size:0.9rem;'>
                  <strong>Validation:</strong>
                  <span style='font-family:monospace;'>{passed} / {total}</span>
                  checks passed across {len({r['stage'] for r in val_log})} stage(s)
                </div>
                <div style='background:{color};color:#fff;padding:3px 10px;
                            border-radius:4px;font-size:0.75rem;font-weight:600;'>
                  {label}
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section_header(
        "Pipeline",
        "State of each stage with one-click rerun. Reruns overwrite the "
        "stage's existing output."
    )
    stages = db.get_stage_status(scan_run_id) or []
    stages_by_name = {s["stage"]: s for s in stages}

    for stage_key, stage_label in _STAGE_ORDER:
        s = stages_by_name.get(stage_key, {
            "state": "not_started", "progress": None, "message": "",
        })
        cols = st.columns([4, 1])
        with cols[0]:
            pipeline_stage_card(
                stage_label,
                s.get("state", "not_started"),
                s.get("message") or "",
                s.get("progress"),
            )
        with cols[1]:
            disabled = get_orchestrator is None
            if st.button(
                "▶ Rerun",
                key=f"rerun_{stage_key}_{scan_run_id}",
                use_container_width=True,
                disabled=disabled,
                help=f"Re-run the {stage_label.lower()} stage. Overwrites existing output."
                     if not disabled else "Orchestrator not available.",
            ):
                try:
                    with st.spinner(f"Running {stage_label}…"):
                        get_orchestrator().rerun_stage(scan_run_id, stage_key)
                    st.success(f"{stage_label} rerun complete.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Rerun failed: {exc}")

    with st.expander("Full stage status table", expanded=False):
        stage_status_table(stages)

    section_header("Per-practice summary")
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for f in findings:
        grouped.setdefault(f["practice"], []).append(f)

    if not grouped:
        st.info("No findings yet — detection may still be running.")
        return

    for practice, flist in grouped.items():
        practice_summary_card(practice, summarise_findings(flist))

def render_files(db: AuditDatabase, scan_run_id: int) -> None:
    candidates = db.get_all_candidates(scan_run_id)
    if not candidates:
        st.info("No candidate files for this scan run yet.")
        return

    section_header("Candidate files",
                   "All files that survived the keyword pre-filter.")

    statuses = sorted({c["status"] for c in candidates})
    selected_statuses = st.multiselect(
        "Filter by status", options=statuses, default=statuses,
        help="`selected` = passed filter, `validated` = passed file checks, "
             "`complete` = analyzed, `rejected` = failed file validation.",
    )
    filtered = [c for c in candidates if c["status"] in selected_statuses]

    rows = []
    for c in filtered:
        rows.append({
            "File":           Path(c["file_path"]).name,
            "Type":           c.get("file_type") or "—",
            "Status":         c["status"],
            "Keyword score":  c.get("keyword_score", 0),
            "Rejection":      c.get("rejection_reason") or "",
            "Full path":      c["file_path"],
        })

    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Keyword score": st.column_config.NumberColumn(width="small"),
            "Status":        st.column_config.TextColumn(width="small"),
            "Type":          st.column_config.TextColumn(width="small"),
            "Full path":     st.column_config.TextColumn(width="medium"),
        },
    )
    st.caption(f"Showing {len(filtered)} of {len(candidates)} files.")


def render_findings(
    db: AuditDatabase,
    scan_run_id: int,
    get_orchestrator: Optional[Callable] = None,
) -> None:
    findings = db.get_findings(scan_run_id)
    if not findings:
        st.info("No findings yet — detection may still be running.")
        return

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for f in findings:
        grouped.setdefault(f["practice"], []).append(f)

    practice_names = list(grouped.keys())
    selected_practice = st.selectbox(
        "Practice to inspect",
        options=practice_names,
        index=0 if practice_names else None,
    )
    if not selected_practice:
        return

    findings_for_practice = grouped[selected_practice]
    summary = summarise_findings(findings_for_practice)
    practice_summary_card(selected_practice, summary)

    filter_choice = st.radio(
        "Show:",
        options=["All", "✓ Supported only", "✗ Not supported only", "⚠ Parse warnings only"],
        horizontal=True,
    )

    def _matches(f: Dict[str, Any]) -> bool:
        if filter_choice == "✓ Supported only":
            return f.get("supported", False)
        if filter_choice == "✗ Not supported only":
            return not f.get("supported", False)
        if filter_choice == "⚠ Parse warnings only":
            return bool(f.get("parse_warning"))
        return True

    visible = [f for f in findings_for_practice if _matches(f)]
    if not visible:
        st.caption("_No findings match the current filter._")
        return

    for f in visible:
        sup = f.get("supported", False)
        warn = bool(f.get("parse_warning"))
        prefix = "⚠" if warn else ("✓" if sup else "✗")
        title = f"{prefix}  {Path(f['file_path']).name}"
        with st.expander(title, expanded=warn or sup):
            # Show full path as a caption inside the expander
            st.caption(f"Path: `{f['file_path']}`")
            evidence_card({
                "file":          f["file_path"],
                "supported":     f["supported"],
                "evidence":      f.get("evidence", []),
                "assets":        f.get("assets", {}),
                "parse_warning": f.get("parse_warning"),
                "was_extracted": f.get("was_extracted", False),
            })

            if get_orchestrator is not None:
                st.markdown("---")
                btn_cols = st.columns([3, 2])
                with btn_cols[1]:
                    btn_key = f"reanalyze_{scan_run_id}_{f['practice']}_{f['file_path']}"
                    if st.button(
                        "🔄 Re-analyze this file",
                        key=btn_key,
                        use_container_width=True,
                        help="Re-runs detection on this file for ALL practices. "
                             "Existing findings for this file are overwritten."
                    ):
                        try:
                            with st.spinner(f"Re-analyzing {Path(f['file_path']).name}…"):
                                out = get_orchestrator().reanalyze_file(
                                    scan_run_id, f["file_path"]
                                )
                            st.success(
                                f"Re-analyzed (replaced {out.get('rows_replaced', 0)} old findings)."
                            )
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Re-analyze failed: {exc}")


def render_validation_log(db: AuditDatabase, scan_run_id: int) -> None:
    log = db.get_validation_log(scan_run_id)
    if not log:
        st.info("No validation checks have run yet.")
        return

    section_header(
        "Validation log",
        "Post-stage sanity checks: did the LLM follow the contract?",
    )

    total = len(log)
    passed = sum(1 for r in log if r["passed"])
    cols = st.columns(3)
    cols[0].metric("Total checks", total)
    cols[1].metric("Passed", passed)
    cols[2].metric("Failed", total - passed)

    st.markdown("---")

    for check in log:
        validation_check_row(check)


def render_raw_report(db: AuditDatabase, scan_run_id: int) -> None:
    from database.db import STAGE_REPORT
    cp = db.load_checkpoint(scan_run_id, STAGE_REPORT)
    if not cp:
        st.info(
            "No report has been generated yet for this scan run. "
            "The report is generated automatically at the end of a full pipeline run."
        )
        return

    paths = cp["payload"].get("report_paths", {})
    html_path = paths.get("html")
    json_path = paths.get("json")

    cols = st.columns(2)
    if html_path:
        cols[0].markdown(f"**HTML:** `{html_path}`")
    if json_path:
        cols[1].markdown(f"**JSON:** `{json_path}`")

    if html_path and Path(html_path).exists():
        section_header("HTML report (embedded)")
        html_text = Path(html_path).read_text(encoding="utf-8")
        st.components.v1.html(html_text, height=1600, scrolling=True)
    else:
        st.warning("HTML report file not found on disk.")
