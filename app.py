"""
app.py
------
Streamlit UI for the RL Auditor.

Run with:
    streamlit run app.py

The app:
  * Sidebar — pick / clone a repo, configure model + max files, start scans,
              and load any past scan from the DB.
  * Tabs    — Overview, Files, Findings (per-practice, with evidence cards
              showing line numbers + code + LLM explanation), Validation log,
              Raw HTML report.

Scans run in a background thread so the UI can poll the DB for live
progress without freezing. Zero changes to the core auditor — we read
from `stage_status` (which every server updates per-iteration) and the
findings table as detection populates it.
"""

from __future__ import annotations
import os
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

# Make sure the v3 package root is importable when running `streamlit run app.py`
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from database.db import AuditDatabase
from src.llm import OllamaClient
from agents.orchestrator_agent import OrchestratorAgent

from detectors.sim_async_parallel_conversation import (
    create_python_rl_sim_async_parallel_conversation,
)
from detectors.real_world_shadow_conversation import (
    create_python_rl_real_world_shadow_conversation,
)
from detectors.hybrid_sim_to_real_conversation import (
    create_python_rl_hybrid_sim_to_real_conversation,
)
from detectors.offline_batch_conversation import (
    create_python_rl_offline_batch_conversation,
)
from detectors.human_in_the_loop_conversation import (
    create_python_rl_human_in_the_loop_conversation,
)
from detectors.league_based_conversation import (
    create_python_rl_league_based_conversation,
)
from detectors.preference_based_conversation import (
    create_python_rl_preference_based_conversation,
)

from ui.scan_runner import ScanThreadState, start_scan, is_running
from ui.pages import (
    render_overview,
    render_files,
    render_findings,
    render_validation_log,
    render_raw_report,
)
from ui.components import state_badge


# ── Detectors registry — mirrors main.py ──────────────────────────────
DETECTORS = {
    "Simulation-Based (Async/Parallel)": create_python_rl_sim_async_parallel_conversation,
    "Real-World (Shadow Mode)":          create_python_rl_real_world_shadow_conversation,
    "Hybrid (Sim-to-Real)":              create_python_rl_hybrid_sim_to_real_conversation,
    "Offline (Batch)":                   create_python_rl_offline_batch_conversation,
    "Human-in-the-Loop":                 create_python_rl_human_in_the_loop_conversation,
    "League-Based Curriculum":           create_python_rl_league_based_conversation,
    "Preference-Data Collection (Pairwise Comparison)": create_python_rl_preference_based_conversation,
}


# ── Page config + global CSS ──────────────────────────────────────────
st.set_page_config(
    page_title="RL Auditor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
      h1 { color: #e94560; }
      .stTabs [data-baseweb="tab-list"] { gap: 4px; }
      .stTabs [data-baseweb="tab"] {
        background: #16213e; border-radius: 4px; padding: 0.4rem 1rem;
        color: #ccc;
      }
      .stTabs [aria-selected="true"] { background: #e94560 !important; color:#fff !important; }
      div[data-testid="stMetricValue"] { color: #e94560; }
      .stProgress > div > div > div > div { background-color: #e94560; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session state setup ───────────────────────────────────────────────
if "scan_state" not in st.session_state:
    st.session_state.scan_state = ScanThreadState()

if "selected_run_id" not in st.session_state:
    st.session_state.selected_run_id = None

if "settings" not in st.session_state:
    st.session_state.settings = {
        "model":       config.OLLAMA_MODEL,
        "base_url":    config.OLLAMA_BASE_URL,
        "max_files":   config.MAX_FILES,
        "db_path":     str(config.DB_PATH),
        "reports_dir": config.REPORTS_DIR,
    }


# ── Helpers ───────────────────────────────────────────────────────────
@st.cache_resource
def get_db(db_path: str) -> AuditDatabase:
    """Cached so we don't re-open the DB on every rerender."""
    return AuditDatabase(db_path)


def get_orchestrator() -> OrchestratorAgent:
    """Build a fresh orchestrator each time. Cheap — agents are stateless."""
    s = st.session_state.settings
    db = get_db(s["db_path"])
    llm = OllamaClient(
        base_url=s["base_url"],
        model=s["model"],
        max_tokens=config.OLLAMA_MAX_TOKENS,
        timeout=config.OLLAMA_TIMEOUT,
    )
    return OrchestratorAgent(
        db=db, llm_client=llm, detectors=DETECTORS,
        max_files=s["max_files"], reports_dir=s["reports_dir"],
        verbose=False,
    )


def list_past_scans(db: AuditDatabase, limit: int = 50) -> list[dict]:
    with db._conn() as conn:
        rows = conn.execute(
            "SELECT id, repo_root, started_at, status, total_candidates "
            "FROM scan_runs ORDER BY started_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def clone_github_repo(github_url: str, dest_root: str = "/tmp") -> tuple[bool, str]:
    """git clone helper. Returns (success, message_or_path)."""
    if not github_url.strip():
        return False, "Please enter a URL."
    name = github_url.rstrip("/").split("/")[-1].replace(".git", "")
    if not name:
        return False, "Could not derive a repo name from the URL."
    dest = os.path.join(dest_root, name)
    if os.path.isdir(dest):
        return True, dest  # Already cloned
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", github_url, dest],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            return False, f"git clone failed: {result.stderr.strip()}"
        return True, dest
    except FileNotFoundError:
        return False, "git is not installed on this machine."
    except subprocess.TimeoutExpired:
        return False, "git clone timed out after 120 seconds."
    except Exception as exc:
        return False, f"git clone error: {exc}"


# ── Sidebar ───────────────────────────────────────────────────────────
def render_sidebar() -> None:
    st.sidebar.title("🔍 RL Auditor")
    st.sidebar.caption("Agentic LLM-based RL practice detector")

    # ── Repo source ─────────────────────────────────────────────────
    st.sidebar.markdown("### 📁 Repository")

    source = st.sidebar.radio(
        "Source",
        options=["Local path", "Clone from GitHub", "Recent scans"],
        label_visibility="collapsed",
    )

    repo_path = None

    if source == "Local path":
        repo_path = st.sidebar.text_input(
            "Path to repo on disk",
            placeholder="/path/to/your/repo",
            help="Absolute path to the repository you want to scan.",
        )
        if repo_path and not os.path.isdir(repo_path):
            st.sidebar.warning(f"Not a directory: `{repo_path}`")

    elif source == "Clone from GitHub":
        url = st.sidebar.text_input(
            "GitHub URL",
            placeholder="https://github.com/DLR-RM/stable-baselines3",
        )
        if st.sidebar.button("📥 Clone", use_container_width=True):
            with st.spinner("Cloning..."):
                ok, msg = clone_github_repo(url)
            if ok:
                st.session_state.cloned_repo_path = msg
                st.sidebar.success(f"Cloned to `{msg}`")
            else:
                st.sidebar.error(msg)
        if "cloned_repo_path" in st.session_state:
            repo_path = st.session_state.cloned_repo_path
            st.sidebar.caption(f"Will scan: `{repo_path}`")

    elif source == "Recent scans":
        db = get_db(st.session_state.settings["db_path"])
        past = list_past_scans(db, limit=20)
        if not past:
            st.sidebar.info("No past scans yet.")
        else:
            options = {
                f"#{p['id']} · {Path(p['repo_root']).name} · {p['status']}": p['id']
                for p in past
            }
            choice = st.sidebar.selectbox(
                "Load a past scan", options=list(options.keys()),
                label_visibility="collapsed",
            )
            if st.sidebar.button("📂 Open", use_container_width=True):
                st.session_state.selected_run_id = options[choice]
                st.rerun()

    # ── Settings ──────────────────────────────────────────────────────
    st.sidebar.markdown("### ⚙️ Settings")
    s = st.session_state.settings
    s["model"]      = st.sidebar.text_input("Ollama model",        s["model"])
    s["base_url"]   = st.sidebar.text_input("Ollama URL",          s["base_url"])
    s["max_files"]  = st.sidebar.slider(    "Max files (0 = all)", 0, 500, s["max_files"])

    with st.sidebar.expander("Advanced", expanded=False):
        s["db_path"]     = st.text_input("DB path",      s["db_path"])
        s["reports_dir"] = st.text_input("Reports dir",  s["reports_dir"])
        st.caption(f"Practices: {len(DETECTORS)}")
        for name in DETECTORS:
            st.caption(f"  • {name}")

    # ── Run button ────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    can_run = repo_path and os.path.isdir(repo_path) and not is_running(st.session_state.scan_state)

    if st.sidebar.button(
        "▶ Run new scan",
        use_container_width=True,
        type="primary",
        disabled=not can_run,
    ):
        if not repo_path:
            st.sidebar.error("Pick a repo first.")
        else:
            orch = get_orchestrator()
            start_scan(st.session_state.scan_state, orch, repo_path)
            st.session_state.selected_run_id = None  # will be set when worker finishes
            st.rerun()

    if is_running(st.session_state.scan_state):
        st.sidebar.warning("⏳ Scan in progress — UI will refresh.")

    # ── Actions on the currently-loaded scan ──────────────────────────
    # Shown only when a past scan is loaded (selected_run_id is set).
    selected = st.session_state.get("selected_run_id")
    if selected is not None and not is_running(st.session_state.scan_state):
        from ui.sidebar_actions import render_scan_actions
        db_for_actions = get_db(st.session_state.settings["db_path"])
        render_scan_actions(
            db=db_for_actions,
            scan_run_id=selected,
            get_orchestrator=get_orchestrator,
            scan_state=st.session_state.scan_state,
        )

    # MCP tool catalog at the bottom (collapsible)
    with st.sidebar.expander("🧰 MCP tool catalog", expanded=False):
        try:
            tools = get_orchestrator().dispatcher.list_all_tools()
            for t in tools:
                st.caption(f"`{t['server']}.{t['name']}`")
        except Exception as exc:
            st.caption(f"_Could not load tools: {exc}_")


# ── Main area ─────────────────────────────────────────────────────────
def render_running_scan(repo_root: str) -> None:
    """Live-updating progress view while a scan is in flight."""
    st.markdown(
        f"## ⏳ Scanning `{repo_root}`",
    )
    state = st.session_state.scan_state

    # We may not have a scan_run_id assigned yet on the first ticks.
    db = get_db(st.session_state.settings["db_path"])

    # The current scan is the most recent one in the DB if we don't have an
    # ID in state yet (the worker thread will set state.scan_run_id once
    # orchestrator.run() has created the scan).
    scan_run_id = state.scan_run_id
    if scan_run_id is None:
        past = list_past_scans(db, limit=1)
        if past:
            scan_run_id = past[0]["id"]

    progress_container = st.empty()
    stages_container   = st.empty()
    findings_container = st.empty()

    POLL_INTERVAL = 1.5  # seconds

    while is_running(state):
        # Stage status
        stages = db.get_stage_status(scan_run_id) if scan_run_id else []
        if stages:
            with stages_container.container():
                st.markdown("### Stage progress")
                # Overall progress bar across all stages
                stage_order = ["repository_scan", "file_selection", "validation",
                               "detection", "report_generation"]
                stages_by_name = {s["stage"]: s for s in stages}
                complete_count = sum(
                    1 for stage in stage_order
                    if stages_by_name.get(stage, {}).get("state") == "complete"
                )
                st.progress(complete_count / len(stage_order),
                            text=f"{complete_count}/{len(stage_order)} stages complete")

                # Per-stage badges
                cols = st.columns(len(stage_order))
                for col, stage_name in zip(cols, stage_order):
                    s = stages_by_name.get(stage_name, {"state": "not_started", "message": ""})
                    col.markdown(
                        f"<div style='text-align:center;font-size:0.8rem;color:#aaa;'>{stage_name}</div>"
                        f"<div style='text-align:center;margin-top:4px;'>{state_badge(s['state'])}</div>"
                        f"<div style='text-align:center;color:#888;font-size:0.7rem;margin-top:6px;height:2.2rem;'>"
                        f"{(s.get('message') or '')[:50]}</div>",
                        unsafe_allow_html=True,
                    )

        # Live findings count
        if scan_run_id:
            findings = db.get_findings(scan_run_id)
            with findings_container.container():
                if findings:
                    sup = sum(1 for f in findings if f["supported"])
                    warn = sum(1 for f in findings if f.get("parse_warning"))
                    cols = st.columns(3)
                    cols[0].metric("Findings so far", len(findings))
                    cols[1].metric("Supported",       sup)
                    cols[2].metric("Parse warnings",  warn)

        time.sleep(POLL_INTERVAL)
        st.rerun()


def render_results(scan_run_id: int) -> None:
    db = get_db(st.session_state.settings["db_path"])
    run = db.get_scan_run(scan_run_id)
    if not run:
        st.error(f"Scan run #{scan_run_id} not found in the DB.")
        return

    # Top header
    st.markdown(f"## 🔍 Scan #{scan_run_id}")
    st.caption(f"Repo: `{run['repo_root']}` &middot; Status: **{run.get('status', '—')}**")

    # Tabs — Tools is added as the rightmost tab
    (
        tab_overview, tab_files, tab_findings,
        tab_validation, tab_report, tab_tools,
    ) = st.tabs([
        "📊 Overview", "📁 Files", "🎯 Findings",
        "✅ Validation log", "📄 Raw report", "🧰 Tools",
    ])

    with tab_overview:
        render_overview(db, scan_run_id, get_orchestrator=get_orchestrator)
    with tab_files:
        render_files(db, scan_run_id)
    with tab_findings:
        render_findings(db, scan_run_id, get_orchestrator=get_orchestrator)
    with tab_validation:
        render_validation_log(db, scan_run_id)
    with tab_report:
        render_raw_report(db, scan_run_id)
    with tab_tools:
        from ui.tools_tab import render_tools_tab
        render_tools_tab(get_orchestrator().dispatcher, current_scan_id=scan_run_id)


def render_empty_state() -> None:
    st.markdown("# 🔍 RL Auditor")
    st.markdown(
        """
        Welcome. This tool scans a repository for **reinforcement-learning
        data-collection practices** and produces a structured report with
        evidence (file path, line number, code snippet, LLM explanation).

        **To get started:**
        1. Pick a repository in the sidebar (local path, GitHub clone, or pick a past scan).
        2. Adjust settings if you like (model, max files).
        3. Click **▶ Run new scan**.

        Live progress will appear here while the scan runs. When it
        finishes, full results show up in tabs.
        """
    )
    st.info(
        "**Detected practices in this build:**  \n"
        + "  \n".join(f"• {name}" for name in DETECTORS.keys()),
        icon="🎯",
    )


# ── Main ──────────────────────────────────────────────────────────────
def main() -> None:
    render_sidebar()
    state = st.session_state.scan_state

    # If the sidebar flagged a rerun-detection request, dispatch it now.
    # (Two-step pattern so the sidebar's st.rerun() can disable buttons
    # before the long-running thread actually starts.)
    if "rerun_target_id" in st.session_state:
        from ui.sidebar_actions import trigger_rerun_detection_if_requested
        trigger_rerun_detection_if_requested(get_orchestrator, state)

    # 1) If a scan is in flight, show the live view.
    if is_running(state):
        render_running_scan(state.repo_root or "")
        return

    # 2) If a scan just finished, jump to its results.
    if state.finished and state.result:
        st.session_state.selected_run_id = state.scan_run_id
        scan_run_id = state.scan_run_id
        if state.error:
            st.error(f"Scan failed: {state.error}")
            if state.error_trace:
                with st.expander("Traceback"):
                    st.code(state.error_trace)
        state.finished = False  # latch — don't keep firing
        render_results(scan_run_id)
        return

    # 3) If a scan errored out before producing a result.
    if state.finished and state.error:
        st.error(f"Scan failed: {state.error}")
        if state.error_trace:
            with st.expander("Traceback"):
                st.code(state.error_trace)
        state.finished = False
        # Don't return — fall through to whatever is selected.

    # 4) Otherwise, render the selected scan if any.
    scan_run_id = st.session_state.selected_run_id
    if scan_run_id is None:
        render_empty_state()
        return

    render_results(scan_run_id)


if __name__ == "__main__":
    main()
