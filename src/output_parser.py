from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Tuple

_NOT_FOUND_SENTINEL = "no evidence found"

_JSON_FENCE_RE   = re.compile(r"⁠ json\s*([\s\S]*?) ⁠", re.IGNORECASE)
_CODE_FENCE_RE   = re.compile(r"⁠ (?:python|java|yaml)?\s*([\s\S]*?) ⁠", re.IGNORECASE)
_LEADING_NUM_RE  = re.compile(r"^\s*(\d{1,6})\s*[:\-\.]\s*")


def _first_line_number(snippet: str) -> int | None:
    """
    Detect a leading line number on the first non-blank line of a snippet
    (we add them in detection_server before sending to the LLM).
    """
    for raw in snippet.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _LEADING_NUM_RE.match(line)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
        return None
    return None


def _normalise_evidence_item(item: Any) -> Dict[str, Any]:
    """Coerce any evidence-like object into the canonical dict shape."""
    if isinstance(item, dict):
        return {
            "line_number": item.get("line_number"),
            "code_snippet": str(item.get("code_snippet", "")).strip(),
            "explanation": str(item.get("explanation", "")).strip(),
        }

    snippet = str(item).strip()
    return {
        "line_number": _first_line_number(snippet),
        "code_snippet": snippet,
        "explanation": "",
    }


def _try_json_parse(text: str) -> Dict[str, Any] | None:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    m = _JSON_FENCE_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    return None


def parse_llm_response(file_path: str, raw_response: str) -> Dict[str, Any]:
    """
    Convert a raw LLM string into a structured finding dict:

        {
          "file": str,
          "supported": bool,
          "evidence": [ {line_number, code_snippet, explanation}, ... ],
          "assets": dict,
          "raw_response": str,
          "parse_warning": str | None
        }
    """
    stripped = (raw_response or "").strip()

    if stripped.lower().startswith(_NOT_FOUND_SENTINEL):
        return {
            "file":          file_path,
            "supported":     False,
            "evidence":      [],
            "assets":        {},
            "raw_response":  stripped,
            "parse_warning": None,
        }

    obj = _try_json_parse(stripped)
    if obj is not None:
        supported = bool(obj.get("supported", False))
        raw_ev = obj.get("evidence") or []
        evidence: List[Dict[str, Any]] = [
            _normalise_evidence_item(i) for i in raw_ev if i
        ]
        assets = obj.get("assets") or {}
        if not isinstance(assets, dict):
            assets = {"notes": str(assets)}
        return {
            "file":          file_path,
            "supported":     supported,
            "evidence":      evidence,
            "assets":        assets,
            "raw_response":  stripped,
            "parse_warning": None,
        }

    had_fences = bool(_CODE_FENCE_RE.search(stripped))
    return {
        "file":          file_path,
        "supported":     False,
        "evidence":      [],
        "assets":        {},
        "raw_response":  stripped,
        "parse_warning": (
            "LLM did not return a valid JSON object"
            + (" (returned fenced code instead)" if had_fences else "")
            + "; verdict defaulted to not-supported and flagged for review."
        ),
    }


def summarise_findings(findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-file findings into a scan-level summary."""
    detected     = [f for f in findings if f.get("supported")]
    not_detected = [f for f in findings if not f.get("supported")]
    warnings     = [f for f in findings if f.get("parse_warning")]
    return {
        "total_files_scanned":    len(findings),
        "files_with_evidence":    len(detected),
        "files_without_evidence": len(not_detected),
        "parse_warnings":         len(warnings),
        "compliance_detected":    len(detected) > 0,
    }