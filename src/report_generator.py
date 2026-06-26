"""
src/report_generator.py
-----------------------
Transform scan results into JSON and HTML reports.

The report schema (v3) for each finding:
  {
    "file":          str,
    "supported":     bool,
    "evidence":      [ {line_number, code_snippet, explanation}, ... ],
    "assets":        dict,      # LLM-provided extra feedback
    "parse_warning": str | None,
    "was_extracted": bool
  }

HTML rendering shows, per finding:
  * Supported: YES / NO badge
  * Evidence table (line number, snippet, explanation)
  * LLM assets / notes
  * Parse warning (if any)
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _evidence_html(evidence: List[Dict[str, Any]]) -> str:
    if not evidence:
        return "<p class='no-evidence'>No evidence recorded.</p>"
    rows = []
    for item in evidence:
        line = item.get("line_number")
        line_html = f"L{line}" if line is not None else "—"
        rows.append(
            f"<tr>"
            f"<td class='line'>{_escape(line_html)}</td>"
            f"<td><pre><code>{_escape(item.get('code_snippet', ''))}</code></pre></td>"
            f"<td class='expl'>{_escape(item.get('explanation', ''))}</td>"
            f"</tr>"
        )
    return (
        "<table class='evidence-table'>"
        "<thead><tr><th>Line</th><th>Code</th><th>Explanation</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _assets_html(assets: Dict[str, Any]) -> str:
    if not assets:
        return ""
    pretty = json.dumps(assets, indent=2)
    return (
        f"<details class='assets'><summary>LLM notes / assets</summary>"
        f"<pre><code>{_escape(pretty)}</code></pre></details>"
    )


class ReportGenerator:
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_json(
        self, scan_result: Dict[str, Any], filename: str = "report.json",
    ) -> Path:
        payload = {"generated_at": _now_iso(), **scan_result}
        dest = self.output_dir / filename
        dest.write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        print(f"[ReportGenerator] JSON report → {dest}")
        return dest

    def save_html(
        self, scan_result: Dict[str, Any], filename: str = "report.html",
    ) -> Path:
        html = self._build_html(scan_result)
        dest = self.output_dir / filename
        dest.write_text(html, encoding="utf-8")
        print(f"[ReportGenerator] HTML report → {dest}")
        return dest

    def save_all(
        self, scan_result: Dict[str, Any], stem: str = "scan",
    ) -> Dict[str, Path]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return {
            "json": self.save_json(scan_result, f"{stem}_{ts}.json"),
            "html": self.save_html(scan_result, f"{stem}_{ts}.html"),
        }

    def _build_html(self, scan_result: Dict[str, Any]) -> str:
        repo = scan_result.get("repo_root", "N/A")
        ts   = _now_iso()
        results_by_practice: Dict[str, Any] = scan_result.get("results_by_practice", {})

        # ── Summary cards ─────────────────────────────────────────────
        summary_cards = ""
        for practice_name, data in results_by_practice.items():
            s = data["summary"]
            detected = s.get("compliance_detected", False)
            color    = "#2ecc71" if detected else "#e74c3c"
            badge    = "DETECTED" if detected else "NOT DETECTED"
            summary_cards += (
                f"<div class='card'>"
                f"<div class='card-title'>{_escape(practice_name)}</div>"
                f"<div class='badge' style='background:{color}'>{badge}</div>"
                f"<div class='card-stats'>"
                f"{s.get('files_with_evidence', 0)} / "
                f"{s.get('total_files_scanned', 0)} files "
                f"&nbsp;|&nbsp; {s.get('parse_warnings', 0)} warnings"
                f"</div>"
                f"</div>"
            )

        # ── Per-practice detail sections ─────────────────────────────
        detail_sections = ""
        for practice_name, data in results_by_practice.items():
            findings: List[Dict[str, Any]] = data["findings"]
            s = data["summary"]
            detected = s.get("compliance_detected", False)
            header_color = "#2ecc71" if detected else "#e74c3c"

            finding_rows = ""
            for f in findings:
                # Accept either the new 'supported' or legacy 'detected' key.
                is_sup = f.get("supported", f.get("detected", False))
                status_color = "#2ecc71" if is_sup else "#e74c3c"
                status_label = "✓ Supported" if is_sup else "✗ Not supported"
                warn = f.get("parse_warning") or ""
                warn_html = (
                    f"<p class='warning'>⚠ {_escape(warn)}</p>" if warn else ""
                )
                extr_html = (
                    "<span class='tag'>extracted</span>"
                    if f.get("was_extracted") else ""
                )
                evidence_html = _evidence_html(f.get("evidence", []))
                assets_html   = _assets_html(f.get("assets", {}))

                finding_rows += (
                    f"<details>"
                    f"<summary>"
                    f"<span style='color:{status_color};font-weight:bold'>"
                    f"{status_label}</span>"
                    f"{extr_html}"
                    f"&nbsp;&nbsp;<code>{_escape(f.get('file', ''))}</code>"
                    f"</summary>"
                    f"{warn_html}"
                    f"<div class='evidence'>{evidence_html}</div>"
                    f"{assets_html}"
                    f"</details>"
                )

            detail_sections += (
                f"<div class='practice-section'>"
                f"<h2 style='border-left: 4px solid {header_color}; "
                f"padding-left: .6rem;'>{_escape(practice_name)}</h2>"
                f"{finding_rows or '<p>No files analysed.</p>'}"
                f"</div>"
            )

        # ── Stage status block (if present) ──────────────────────────
        stage_status_html = ""
        stage_rows = scan_result.get("stage_status") or []
        if stage_rows:
            rows_html = ""
            for s in stage_rows:
                state = s.get("state", "")
                color = {
                    "complete":     "#2ecc71",
                    "partial":      "#f1c40f",
                    "running":      "#3498db",
                    "failed":       "#e74c3c",
                    "not_started":  "#7f8c8d",
                }.get(state, "#7f8c8d")
                rows_html += (
                    f"<tr>"
                    f"<td>{_escape(s.get('stage', ''))}</td>"
                    f"<td><span class='state-badge' style='background:{color}'>"
                    f"{_escape(state)}</span></td>"
                    f"<td>{_escape(s.get('message') or '')}</td>"
                    f"</tr>"
                )
            stage_status_html = (
                "<h2>Stage Status</h2>"
                "<table class='stage-table'><thead><tr>"
                "<th>Stage</th><th>State</th><th>Message</th>"
                "</tr></thead>"
                f"<tbody>{rows_html}</tbody></table>"
            )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>RL Auditor — Multi-Practice Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; background:#1a1a2e; color:#eee; margin:0; padding:2rem; }}
    h1   {{ color:#e94560; }}
    .meta {{ color:#aaa; margin-bottom:1.5rem; font-size:.9rem; }}
    .cards {{ display:flex; flex-wrap:wrap; gap:1rem; margin:1.5rem 0; }}
    .card {{ background:#16213e; border-radius:8px; padding:1.2rem 1.5rem; min-width:220px; }}
    .card-title {{ font-size:.85rem; color:#aaa; margin-bottom:.4rem; }}
    .badge {{ display:inline-block; padding:.25rem .7rem; border-radius:4px;
              color:#fff; font-weight:bold; font-size:.95rem; margin-bottom:.4rem; }}
    .card-stats {{ font-size:.8rem; color:#aaa; }}
    .practice-section {{ margin:2rem 0; }}
    details {{ background:#16213e; border-radius:6px; margin:.5rem 0; padding:.7rem 1rem; }}
    summary {{ cursor:pointer; font-size:.92rem; }}
    pre     {{ background:#0f3460; border-radius:4px; padding:.6rem;
               overflow-x:auto; font-size:.8rem; color:#a8d8ea; margin:0; }}
    .evidence-table {{ width:100%; border-collapse:collapse; margin-top:.6rem; font-size:.85rem; }}
    .evidence-table th, .evidence-table td {{ border:1px solid #2c3e50; padding:.4rem .6rem; vertical-align:top; }}
    .evidence-table th {{ background:#0f3460; color:#a8d8ea; text-align:left; }}
    .evidence-table td.line {{ color:#f1c40f; font-family:monospace; width:50px; }}
    .evidence-table td.expl {{ color:#ddd; max-width:400px; }}
    .no-evidence {{ color:#aaa; font-style:italic; }}
    .warning {{ color:#f39c12; font-size:.83rem; }}
    .evidence {{ margin-top:.5rem; }}
    .tag    {{ background:#444; border-radius:3px; font-size:.72rem; padding:.1rem .4rem; color:#aaa; margin-left:.3rem; }}
    .assets {{ margin-top:.6rem; background:#0f1a2e; border-radius:4px; padding:.4rem .7rem; }}
    .stage-table {{ width:100%; border-collapse:collapse; margin-top:1rem; font-size:.88rem; }}
    .stage-table th, .stage-table td {{ border:1px solid #2c3e50; padding:.4rem .8rem; text-align:left; }}
    .stage-table th {{ background:#0f3460; color:#a8d8ea; }}
    .state-badge {{ display:inline-block; padding:.1rem .5rem; border-radius:3px;
                    color:#fff; font-size:.75rem; font-weight:bold; }}
  </style>
</head>
<body>
  <h1>🔍 RL Data Collection Auditor</h1>
  <p class="meta">
    Repo: <code>{_escape(str(repo))}</code>
    &nbsp;|&nbsp; Generated: {ts}
    &nbsp;|&nbsp; Practices scanned: {len(results_by_practice)}
  </p>

  {stage_status_html}

  <h2>Summary</h2>
  <div class="cards">{summary_cards}</div>

  <h2>Detailed Findings</h2>
  {detail_sections}
</body>
</html>"""
