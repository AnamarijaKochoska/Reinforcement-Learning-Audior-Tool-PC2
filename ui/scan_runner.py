"""
ui/scan_runner.py
-----------------
Runs a scan in a background thread so the Streamlit UI can poll the DB
for live progress without freezing.

We don't modify the core auditor — we just call `OrchestratorAgent.run()`
in a worker thread and have the UI re-read `stage_status` every second.
That works because every server already updates `stage_status` after
each file in the detection loop (see detection_server.run_detection's
inner loop), so live progress is "free."

A simple ScanRunner singleton holds the thread + a result/error slot.
"""

from __future__ import annotations
import threading
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass
class ScanThreadState:
    thread:      Optional[threading.Thread] = None
    scan_run_id: Optional[int] = None
    result:      Optional[Dict[str, Any]] = None
    error:       Optional[str] = None
    error_trace: Optional[str] = None
    started:     bool = False
    finished:    bool = False
    repo_root:   Optional[str] = None


def start_scan(
    state: ScanThreadState,
    orchestrator,
    repo_root: str,
    scan_run_id: Optional[int] = None,
) -> None:
    """Kick off a new background scan. Mutates `state` in place."""
    state.repo_root = repo_root
    state.scan_run_id = scan_run_id
    state.result = None
    state.error = None
    state.error_trace = None
    state.started = True
    state.finished = False

    def _worker():
        try:
            result = orchestrator.run(repo_root=repo_root, scan_run_id=scan_run_id)
            state.result = result
            state.scan_run_id = result.get("scan_run_id")
        except Exception as exc:
            state.error = f"{type(exc).__name__}: {exc}"
            state.error_trace = traceback.format_exc()
        finally:
            state.finished = True

    state.thread = threading.Thread(target=_worker, daemon=True)
    state.thread.start()


def is_running(state: ScanThreadState) -> bool:
    return state.started and not state.finished


def reset(state: ScanThreadState) -> None:
    """Reset state, but DON'T kill an in-flight thread (let it complete)."""
    state.started = False
    state.finished = False
    state.result = None
    state.error = None
    state.error_trace = None
    state.scan_run_id = None
    state.repo_root = None
    state.thread = None
