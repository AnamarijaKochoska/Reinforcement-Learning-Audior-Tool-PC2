from __future__ import annotations
import streamlit as st

from database.db import AuditDatabase
from ui.scan_runner import is_running, start_scan
from ui.components import overwrite_notice


def render_scan_actions(
    db: AuditDatabase,
    scan_run_id: int,
    get_orchestrator,
    scan_state,
) -> None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔁 Actions on this scan")
    st.sidebar.caption(f"Loaded: scan #{scan_run_id}")

    if st.sidebar.button(
        "📄 Regenerate report",
        use_container_width=True,
        disabled=is_running(scan_state),
        help="Re-render JSON and HTML from existing findings. No LLM calls.",
    ):
        try:
            with st.spinner("Generating report..."):
                out = get_orchestrator().regenerate_report(scan_run_id)
            st.sidebar.success(
                f"Report regenerated: {list(out.get('report_paths', {}).keys())}"
            )
        except Exception as exc:
            st.sidebar.error(f"Failed: {exc}")

    if st.sidebar.button(
        "✅ Revalidate results",
        use_container_width=True,
        disabled=is_running(scan_state),
        help="Re-run the three post-stage sanity checks on existing findings.",
    ):
        try:
            with st.spinner("Validating..."):
                out = get_orchestrator().revalidate_results(scan_run_id)
            if out.get("overall_passed"):
                st.sidebar.success("All checks passed.")
            else:
                st.sidebar.warning("Some checks failed — see Validation log tab.")
        except Exception as exc:
            st.sidebar.error(f"Failed: {exc}")

    rerun_disabled = is_running(scan_state)
    if rerun_disabled:
        st.sidebar.caption("_Rerun detection unavailable while a scan is running._")
    else:
        overwrite_notice(
            "Reruns wipe and rewrite all findings for this scan."
        )
    if st.sidebar.button(
        "🔄 Rerun detection (slow)",
        use_container_width=True,
        disabled=rerun_disabled,
        help="Wipes all existing findings, then re-runs every practice "
             "against every validated file. Same speed as the original scan.",
    ):
        run = db.get_scan_run(scan_run_id)
        if not run:
            st.sidebar.error("Scan run not found.")
            return
        
        st.session_state["rerun_target_id"] = scan_run_id
        st.sidebar.info("Detection rerun queued — see main panel.")
        st.rerun()

    stages = db.get_stage_status(scan_run_id) or []
    incomplete = [
        s for s in stages if s["state"] not in ("complete",)
    ]
    if incomplete:
        st.sidebar.markdown("")
        st.sidebar.caption(
            f"_{len(incomplete)} stage(s) not complete — resume to finish them._"
        )
        if st.sidebar.button(
            "▶ Resume from incomplete",
            use_container_width=True,
            disabled=is_running(scan_state),
            help="Re-run any pipeline stages that aren't marked 'complete' "
                 "for this scan, in order.",
        ):
            run = db.get_scan_run(scan_run_id)
            if not run:
                st.sidebar.error("Scan run not found.")
                return
            orch = get_orchestrator()
            start_scan(scan_state, orch, run["repo_root"], scan_run_id=scan_run_id)
            st.rerun()


def trigger_rerun_detection_if_requested(
    get_orchestrator,
    scan_state,
) -> None:
    target = st.session_state.pop("rerun_target_id", None)
    if target is None:
        return
    orch = get_orchestrator()

    import threading
    def _worker():
        try:
            scan_state.result = orch.rerun_detection(target)
            scan_state.scan_run_id = target
        except Exception as exc:
            import traceback
            scan_state.error = f"{type(exc).__name__}: {exc}"
            scan_state.error_trace = traceback.format_exc()
        finally:
            scan_state.finished = True
    scan_state.repo_root = None  # we're not running a fresh repo
    scan_state.scan_run_id = target
    scan_state.result = None
    scan_state.error = None
    scan_state.error_trace = None
    scan_state.started = True
    scan_state.finished = False
    t = threading.Thread(target=_worker, daemon=True)
    scan_state.thread = t
    t.start()
