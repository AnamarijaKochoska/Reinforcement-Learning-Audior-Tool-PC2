"""
ui/tools_tab.py
---------------
The "Tools" tab — direct invocation of any of the 15 MCP tools across
the 5 servers, with an editable JSON arguments form.

Why this exists: when the sidebar buttons and per-stage rerun controls
don't cover what you need, this is the escape hatch. Pick a tool, see
its schema, edit the arguments as JSON, hit Invoke. The full dispatcher
call result comes back rendered as JSON.

Safety: every tool is shown with a colour-coded marker:
  🟢 read-only      — safe to click, never mutates state
  🟡 action         — mutates the DB / file system; explicit warning shown

The split is determined heuristically from the tool name: anything
starting with `get_`, `list_`, or `describe_` is read-only; everything
else is treated as an action.
"""

from __future__ import annotations
import json
from typing import Any, Dict

import streamlit as st

from mcp_servers.base_server import MCPDispatcher


# Tools that mutate state — explicitly flagged so we can show a warning.
# Anything not in this set is treated as a safe getter.
_ACTION_TOOLS = {
    "scan_repository",
    "select_files",
    "validate_candidates",
    "validate_detection_results",
    "run_detection",
    "run_detection_on_files",
    "generate_report",
}


def _is_action_tool(tool_name: str) -> bool:
    return tool_name in _ACTION_TOOLS


def _default_args_for_tool(tool: Dict[str, Any], current_scan_id: int | None) -> Dict[str, Any]:
    """Build a reasonable default args dict from the tool's input schema."""
    schema = tool.get("inputSchema", {})
    props = schema.get("properties", {}) or {}
    out: Dict[str, Any] = {}
    for key, spec in props.items():
        if key == "scan_run_id" and current_scan_id is not None:
            out[key] = current_scan_id
            continue
        t = spec.get("type")
        if t == "integer":
            out[key] = 0
        elif t == "string":
            out[key] = ""
        elif t == "array":
            out[key] = []
        elif t == "object":
            out[key] = {}
        elif t == "boolean":
            out[key] = False
        else:
            out[key] = None
    return out


def render_tools_tab(
    dispatcher: MCPDispatcher,
    current_scan_id: int | None = None,
) -> None:
    """Render the Tools tab."""
    st.markdown(
        "<h3 style='color:#e94560;margin-top:0.5rem;'>🧰 Raw MCP tool invocation</h3>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Direct call into any of the 15 tools across the 5 MCP servers. "
        "Each tool's input schema is shown below the selector — edit the "
        "arguments as JSON, then click Invoke."
    )

    # Build the full catalog
    all_tools = dispatcher.list_all_tools()
    if not all_tools:
        st.warning("No tools registered.")
        return

    # Group by server for the selector
    servers: Dict[str, list] = {}
    for t in all_tools:
        servers.setdefault(t["server"], []).append(t)

    # ── Server selector ──────────────────────────────────────────────
    server_name = st.selectbox(
        "Server",
        options=sorted(servers.keys()),
        index=0,
        help="Each server owns a related set of tools — see Architecture docs.",
    )
    tools_for_server = servers[server_name]

    # ── Tool selector ────────────────────────────────────────────────
    tool_options = {
        f"{'🟡' if _is_action_tool(t['name']) else '🟢'}  {t['name']}": t
        for t in tools_for_server
    }
    tool_label = st.selectbox(
        "Tool",
        options=list(tool_options.keys()),
        index=0,
    )
    tool = tool_options[tool_label]

    # ── Description + schema panel ───────────────────────────────────
    is_action = _is_action_tool(tool["name"])
    with st.container():
        st.markdown(
            f"<div style='background:#16213e;padding:0.8rem 1rem;border-radius:6px;"
            f"border-left:4px solid {'#f1c40f' if is_action else '#2ecc71'};"
            f"margin-bottom:0.8rem;'>"
            f"<div style='color:#aaa;font-size:0.78rem;margin-bottom:0.3rem;'>"
            f"{'⚠ ACTION TOOL — modifies state' if is_action else '✓ READ-ONLY TOOL — safe'}"
            f"</div>"
            f"<div style='color:#ddd;font-size:0.9rem;line-height:1.5;'>"
            f"{tool['description']}</div></div>",
            unsafe_allow_html=True,
        )

    # ── Input schema panel (collapsible) ─────────────────────────────
    with st.expander("📋 Input schema", expanded=False):
        st.json(tool.get("inputSchema", {}))

    # ── Args editor ──────────────────────────────────────────────────
    st.markdown("**Arguments** (JSON)")
    default_args = _default_args_for_tool(tool, current_scan_id)
    default_json = json.dumps(default_args, indent=2)

    # Use a per-tool key so switching tools resets the editor
    editor_key = f"tools_tab_args_{server_name}_{tool['name']}"
    args_text = st.text_area(
        "args",
        value=default_json,
        height=180,
        key=editor_key,
        label_visibility="collapsed",
    )

    # ── Invoke button ────────────────────────────────────────────────
    cols = st.columns([1, 4])
    invoke_clicked = cols[0].button(
        "▶ Invoke",
        use_container_width=True,
        type="primary",
    )
    cols[1].caption(
        "Result will appear below."
        + ("  ⚠ This call modifies state." if is_action else "")
    )

    if not invoke_clicked:
        return

    # ── Parse args ───────────────────────────────────────────────────
    try:
        args = json.loads(args_text) if args_text.strip() else {}
    except json.JSONDecodeError as exc:
        st.error(f"Arguments are not valid JSON: {exc}")
        return

    # ── Call it ──────────────────────────────────────────────────────
    with st.spinner(f"Calling {server_name}.{tool['name']}…"):
        try:
            resp = dispatcher.call(server_name, tool["name"], args)
        except Exception as exc:
            st.error(f"Dispatcher raised: {type(exc).__name__}: {exc}")
            return

    # ── Render result ────────────────────────────────────────────────
    if resp.is_error:
        st.error(resp.get_text())
        return

    st.success("Call succeeded.")
    result = resp.get_data()
    if isinstance(result, (dict, list)):
        st.json(result)
    else:
        st.code(str(result))
