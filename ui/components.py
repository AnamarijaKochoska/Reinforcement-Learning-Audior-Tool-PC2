"""
ui/components.py
----------------
Reusable visual components for the Streamlit UI.

Kept in their own module so the page logic in pages.py stays focused on
data flow and layout — and so component styling can be tweaked in one
place.
"""

from __future__ import annotations
import html
import json
from typing import Any, Dict, List

import streamlit as st


# Colour map shared by stage-status badges and practice-detection badges.
STATE_COLORS = {
    "complete":     "#2ecc71",
    "partial":      "#f1c40f",
    "running":      "#3498db",
    "failed":       "#e74c3c",
    "not_started":  "#7f8c8d",
}


def _escape(text: Any) -> str:
    return html.escape(str(text)) if text is not None else ""


def state_badge(state: str) -> str:
    """Inline HTML badge for a stage state."""
    color = STATE_COLORS.get(state, "#7f8c8d")
    return (
        f"<span style='background:{color};color:#fff;padding:2px 10px;"
        f"border-radius:4px;font-size:0.8rem;font-weight:600;"
        f"font-family:monospace;'>{_escape(state)}</span>"
    )


def supported_badge(supported: bool, has_warning: bool = False) -> str:
    """Inline HTML badge for a finding's supported/not-supported status."""
    if has_warning:
        return (
            "<span style='background:#f39c12;color:#fff;padding:2px 10px;"
            "border-radius:4px;font-size:0.8rem;font-weight:600;'>⚠ Parse warning</span>"
        )
    if supported:
        return (
            "<span style='background:#2ecc71;color:#fff;padding:2px 10px;"
            "border-radius:4px;font-size:0.8rem;font-weight:600;'>✓ Supported</span>"
        )
    return (
        "<span style='background:#7f8c8d;color:#fff;padding:2px 10px;"
        "border-radius:4px;font-size:0.8rem;font-weight:600;'>✗ Not supported</span>"
    )


def stage_status_table(rows: List[Dict[str, Any]]) -> None:
    """Render a table of stage status rows with colour badges."""
    if not rows:
        st.info("No stage status yet.")
        return
    table_rows = []
    for r in rows:
        progress = r.get("progress") or {}
        progress_str = (
            f"{progress.get('processed', 0)} / {progress.get('total', '?')}"
            if "processed" in progress else (
                json.dumps(progress) if progress else "—"
            )
        )
        table_rows.append({
            "Stage":    r.get("stage", ""),
            "State":    r.get("state", ""),
            "Progress": progress_str,
            "Message":  r.get("message") or "",
        })
    st.dataframe(
        table_rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "State": st.column_config.TextColumn(width="small"),
        },
    )


def metric_row(metrics: Dict[str, Any]) -> None:
    """Render a row of st.metric widgets from a dict of label → value."""
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label, value)


def evidence_card(finding: Dict[str, Any]) -> None:
    """
    Render one finding (one file × one practice) with all its evidence
    inline. Each evidence item gets a line number, a code block, and the
    LLM's explanation.
    """
    has_warning = bool(finding.get("parse_warning"))
    supported   = bool(finding.get("supported"))

    # Pulled out of the f-string below: older Python versions (<3.12)
    # don't allow backslashes inside f-string expression parts, so the
    # escaped quotes broke the whole module on Python 3.11.
    if finding.get("was_extracted"):
        extracted_tag = (
            "<span style='background:#444;color:#aaa;font-size:0.7rem;"
            "padding:1px 6px;border-radius:3px;'>extracted</span>"
        )
    else:
        extracted_tag = ""

    header_html = (
        f"<div style='display:flex;align-items:center;gap:0.6rem;'>"
        f"{supported_badge(supported, has_warning)}"
        f"<code style='font-size:0.85rem;color:#a8d8ea;'>"
        f"{_escape(finding.get('file', ''))}</code>"
        f"{extracted_tag}"
        f"</div>"
    )
    st.markdown(header_html, unsafe_allow_html=True)

    if has_warning:
        st.warning(f"⚠ {finding['parse_warning']}", icon="⚠️")

    evidence = finding.get("evidence", []) or []
    if not evidence:
        st.caption("No evidence recorded for this file.")
    else:
        for i, item in enumerate(evidence, start=1):
            line = item.get("line_number")
            line_label = f"Line {line}" if line is not None else "—"
            st.markdown(
                f"<div style='margin-top:0.5rem;'>"
                f"<span style='background:#0f3460;color:#f1c40f;"
                f"padding:2px 8px;border-radius:3px;font-family:monospace;"
                f"font-size:0.75rem;'>Evidence #{i} · {line_label}</span></div>",
                unsafe_allow_html=True,
            )
            st.code(item.get("code_snippet", ""), language="python")
            expl = item.get("explanation") or ""
            if expl:
                st.markdown(
                    f"<div style='color:#ccc;font-style:italic;font-size:0.85rem;"
                    f"margin-bottom:0.4rem;padding-left:0.4rem;border-left:2px solid #555;'>"
                    f"💬 {_escape(expl)}</div>",
                    unsafe_allow_html=True,
                )

    assets = finding.get("assets") or {}
    if assets:
        with st.expander("📎 LLM notes / assets", expanded=False):
            st.json(assets)


def practice_summary_card(practice_name: str, summary: Dict[str, Any]) -> None:
    """
    Header card for a single practice's results — shown above its
    detailed findings list.
    """
    detected = summary.get("compliance_detected", False)
    color    = "#2ecc71" if detected else "#e74c3c"
    label    = "DETECTED" if detected else "NOT DETECTED"

    st.markdown(
        f"""
        <div style='background:#16213e;border-left:6px solid {color};
                    padding:1rem 1.4rem;border-radius:4px;margin-bottom:0.8rem;'>
          <div style='display:flex;justify-content:space-between;align-items:center;'>
            <div style='font-size:1.1rem;font-weight:600;color:#eee;'>
              {_escape(practice_name)}
            </div>
            <div style='background:{color};color:#fff;padding:4px 12px;
                        border-radius:4px;font-weight:600;font-size:0.85rem;'>
              {label}
            </div>
          </div>
          <div style='color:#aaa;font-size:0.85rem;margin-top:0.4rem;'>
            {summary.get('files_with_evidence', 0)} / {summary.get('total_files_scanned', 0)}
            files with evidence
            &nbsp;·&nbsp; {summary.get('parse_warnings', 0)} parse warning(s)
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def validation_check_row(check: Dict[str, Any]) -> None:
    """One row in the validation log display."""
    passed = check.get("passed", False)
    icon = "✅" if passed else "❌"
    color = "#2ecc71" if passed else "#e74c3c"
    st.markdown(
        f"""
        <div style='display:flex;gap:0.7rem;align-items:flex-start;
                    padding:0.5rem 0;border-bottom:1px solid #2c3e50;'>
          <div style='font-size:1.2rem;'>{icon}</div>
          <div style='flex:1;'>
            <div style='color:{color};font-weight:600;font-size:0.9rem;'>
              {_escape(check.get('check_name', ''))}
              <span style='color:#888;font-weight:normal;font-size:0.75rem;'>
                · {_escape(check.get('stage', ''))}
              </span>
            </div>
            <div style='color:#ccc;font-size:0.85rem;margin-top:0.2rem;'>
              {_escape(check.get('details', '') or '')}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = "") -> None:
    """Consistent section header style."""
    st.markdown(
        f"<h3 style='color:#e94560;margin-top:1.5rem;margin-bottom:0.4rem;'>"
        f"{_escape(title)}</h3>",
        unsafe_allow_html=True,
    )
    if subtitle:
        st.caption(subtitle)


def overwrite_notice(message: str) -> None:
    """
    Small inline warning shown above a rerun button.

    Used so the user sees the consequence of clicking a destructive action
    without a blocking confirmation dialog — matches the "overwrite is the
    default, no second click required" decision.
    """
    st.markdown(
        f"<div style='background:rgba(241,196,15,0.12);"
        f"border-left:3px solid #f1c40f;padding:.4rem .8rem;"
        f"font-size:0.82rem;color:#f1c40f;margin:0.3rem 0;"
        f"border-radius:3px;'>⚠ {_escape(message)}</div>",
        unsafe_allow_html=True,
    )


def pipeline_stage_card(
    stage_name: str,
    state: str,
    message: str = "",
    progress: Dict[str, Any] | None = None,
) -> None:
    """
    Compact card for one pipeline stage — used in the pipeline view on the
    Overview tab next to the per-stage 'rerun this stage' button.
    """
    color = STATE_COLORS.get(state, "#7f8c8d")
    progress_str = ""
    if progress:
        if "processed" in progress and "total" in progress:
            progress_str = (
                f"{progress.get('processed', 0)} / {progress.get('total', '?')}"
            )
        elif progress:
            progress_str = str(progress)

    st.markdown(
        f"""
        <div style='background:#16213e;padding:0.7rem 0.9rem;
                    border-radius:4px;border-left:3px solid {color};
                    margin-bottom:0.4rem;'>
          <div style='display:flex;justify-content:space-between;align-items:center;'>
            <div style='font-family:monospace;font-size:0.85rem;color:#eee;'>
              {_escape(stage_name)}
            </div>
            {state_badge(state)}
          </div>
          <div style='display:flex;justify-content:space-between;
                      color:#aaa;font-size:0.75rem;margin-top:0.3rem;'>
            <div>{_escape(message or '—')}</div>
            <div style='font-family:monospace;'>{_escape(progress_str)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )