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
    state.started = False
    state.finished = False
    state.result = None
    state.error = None
    state.error_trace = None
    state.scan_run_id = None
    state.repo_root = None
    state.thread = None
