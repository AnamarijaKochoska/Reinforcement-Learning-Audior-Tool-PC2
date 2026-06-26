"""
database/db.py
--------------
Long-term memory layer for the RL Auditor.

Extends v2 schema with:
  * stage_status       — state of each pipeline stage per scan run
                         (not_started | running | partial | complete | failed)
  * checkpoints        — named snapshots of work-in-progress state, so a scan
                         can be resumed or a single downstream tool can be
                         re-invoked without re-running earlier stages
  * validation_log     — one row per validation check run per stage
  * findings.evidence  — now holds structured JSON (line_number, snippet,
                         explanation) rather than raw code strings
  * findings.assets    — extra LLM feedback / notes, stored as JSON

Stage status is the single source of truth callers should query to know
"what has already happened in scan N and what hasn't" — see get_stage_status().
"""

from __future__ import annotations
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "rl_auditor.db"

# ── Canonical stage names ────────────────────────────────────────────
STAGE_REPO_SCAN      = "repository_scan"
STAGE_FILE_SELECTION = "file_selection"
STAGE_VALIDATION     = "validation"
STAGE_DETECTION      = "detection"
STAGE_REPORT         = "report_generation"

ALL_STAGES = [
    STAGE_REPO_SCAN,
    STAGE_FILE_SELECTION,
    STAGE_VALIDATION,
    STAGE_DETECTION,
    STAGE_REPORT,
]

# ── Canonical stage states ───────────────────────────────────────────
STATE_NOT_STARTED = "not_started"
STATE_RUNNING     = "running"
STATE_PARTIAL     = "partial"
STATE_COMPLETE    = "complete"
STATE_FAILED      = "failed"

VALID_STATES = {STATE_NOT_STARTED, STATE_RUNNING, STATE_PARTIAL,
                STATE_COMPLETE, STATE_FAILED}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AuditDatabase:

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self._init_schema()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS scan_runs (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                repo_root        TEXT    NOT NULL,
                started_at       TEXT    NOT NULL,
                completed_at     TEXT,
                status           TEXT    DEFAULT 'running',
                model            TEXT,
                total_candidates INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS candidate_files (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_run_id       INTEGER NOT NULL,
                file_path         TEXT    NOT NULL,
                file_type         TEXT,
                keyword_score     INTEGER DEFAULT 0,
                status            TEXT    DEFAULT 'selected',
                selected_by       TEXT    DEFAULT 'file_selection_agent',
                validated_by      TEXT,
                rejection_reason  TEXT,
                created_at        TEXT    NOT NULL,
                updated_at        TEXT    NOT NULL,
                FOREIGN KEY (scan_run_id) REFERENCES scan_runs(id)
            );

            -- Evidence is now stored as a JSON array of
            --   {"line_number": int, "code_snippet": str, "explanation": str}
            CREATE TABLE IF NOT EXISTS findings (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_run_id    INTEGER NOT NULL,
                file_path      TEXT    NOT NULL,
                practice       TEXT    NOT NULL,
                supported      INTEGER DEFAULT 0,     -- renamed from 'detected'
                evidence       TEXT,                  -- JSON list of evidence objects
                assets         TEXT,                  -- JSON blob of extra LLM feedback
                raw_response   TEXT,
                parse_warning  TEXT,
                was_extracted  INTEGER DEFAULT 0,
                created_at     TEXT    NOT NULL,
                FOREIGN KEY (scan_run_id) REFERENCES scan_runs(id)
            );

            CREATE TABLE IF NOT EXISTS agent_registry (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id       TEXT UNIQUE NOT NULL,
                agent_type     TEXT NOT NULL,
                status         TEXT DEFAULT 'idle',
                current_task   TEXT,
                started_at     TEXT,
                last_heartbeat TEXT
            );

            -- Stage-level state tracking: one row per (scan_run, stage).
            -- Query this to know the state of any step at any time.
            CREATE TABLE IF NOT EXISTS stage_status (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_run_id   INTEGER NOT NULL,
                stage         TEXT    NOT NULL,
                state         TEXT    NOT NULL,  -- not_started|running|partial|complete|failed
                progress      TEXT,              -- JSON, e.g. {"processed": 5, "total": 25}
                message       TEXT,              -- human-readable status
                started_at    TEXT,
                updated_at    TEXT    NOT NULL,
                UNIQUE (scan_run_id, stage),
                FOREIGN KEY (scan_run_id) REFERENCES scan_runs(id)
            );

            -- Checkpoints: named snapshots to resume from or feed into later tools
            CREATE TABLE IF NOT EXISTS checkpoints (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_run_id   INTEGER NOT NULL,
                stage         TEXT    NOT NULL,
                payload       TEXT    NOT NULL,  -- JSON
                created_at    TEXT    NOT NULL,
                UNIQUE (scan_run_id, stage),
                FOREIGN KEY (scan_run_id) REFERENCES scan_runs(id)
            );

            -- Validation log: one row per validation check run per stage
            CREATE TABLE IF NOT EXISTS validation_log (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_run_id   INTEGER NOT NULL,
                stage         TEXT    NOT NULL,
                check_name    TEXT    NOT NULL,
                passed        INTEGER NOT NULL,
                details       TEXT,
                created_at    TEXT    NOT NULL,
                FOREIGN KEY (scan_run_id) REFERENCES scan_runs(id)
            );
            """)

    # ─── scan_runs ──────────────────────────────────────────────────────
    def create_scan_run(self, repo_root: str, model: str) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO scan_runs (repo_root, started_at, model) VALUES (?, ?, ?)",
                (repo_root, _now(), model),
            )
            run_id = cur.lastrowid
            # Pre-seed all stages as not_started so get_stage_status
            # can always return a meaningful answer.
            now = _now()
            for stage in ALL_STAGES:
                conn.execute(
                    """INSERT INTO stage_status
                       (scan_run_id, stage, state, updated_at)
                       VALUES (?, ?, ?, ?)""",
                    (run_id, stage, STATE_NOT_STARTED, now),
                )
            return run_id

    def complete_scan_run(self, scan_run_id: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE scan_runs SET status='complete', completed_at=? WHERE id=?",
                (_now(), scan_run_id),
            )

    def fail_scan_run(self, scan_run_id: int, reason: str = "") -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE scan_runs SET status='failed', completed_at=? WHERE id=?",
                (_now(), scan_run_id),
            )

    def update_scan_run_candidates(self, scan_run_id: int, total: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE scan_runs SET total_candidates=? WHERE id=?",
                (total, scan_run_id),
            )

    def get_scan_run(self, scan_run_id: int) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM scan_runs WHERE id=?", (scan_run_id,),
            ).fetchone()
        return dict(row) if row else None

    # ─── candidate_files ────────────────────────────────────────────────
    def insert_candidate(
        self,
        scan_run_id: int,
        file_path: str,
        file_type: str,
        keyword_score: int,
    ) -> int:
        now = _now()
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO candidate_files
                   (scan_run_id, file_path, file_type, keyword_score,
                    status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, 'selected', ?, ?)""",
                (scan_run_id, file_path, file_type, keyword_score, now, now),
            )
            return cur.lastrowid

    def get_candidates_by_status(
        self, scan_run_id: int, status: str,
    ) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM candidate_files WHERE scan_run_id=? AND status=?",
                (scan_run_id, status),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_candidates(self, scan_run_id: int) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM candidate_files WHERE scan_run_id=?",
                (scan_run_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def update_candidate_status(
        self,
        candidate_id: int,
        status: str,
        validated_by: str | None = None,
        rejection_reason: str | None = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """UPDATE candidate_files
                   SET status=?, validated_by=COALESCE(?, validated_by),
                       rejection_reason=COALESCE(?, rejection_reason),
                       updated_at=?
                   WHERE id=?""",
                (status, validated_by, rejection_reason, _now(), candidate_id),
            )

    # ─── findings ──────────────────────────────────────────────────────
    def insert_finding(
        self,
        scan_run_id: int,
        file_path: str,
        practice: str,
        supported: bool,
        evidence: List[Dict[str, Any]],
        raw_response: str,
        parse_warning: str | None,
        was_extracted: bool,
        assets: Dict[str, Any] | None = None,
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO findings
                   (scan_run_id, file_path, practice, supported, evidence,
                    assets, raw_response, parse_warning, was_extracted, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    scan_run_id, file_path, practice,
                    int(supported),
                    json.dumps(evidence or []),
                    json.dumps(assets or {}),
                    raw_response, parse_warning,
                    int(was_extracted), _now(),
                ),
            )
            return cur.lastrowid

    def get_findings(self, scan_run_id: int) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM findings WHERE scan_run_id=? ORDER BY practice, file_path",
                (scan_run_id,),
            ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["evidence"]      = json.loads(d["evidence"] or "[]")
            d["assets"]        = json.loads(d["assets"] or "{}")
            d["supported"]     = bool(d["supported"])
            d["was_extracted"] = bool(d["was_extracted"])
            results.append(d)
        return results

    def delete_findings_for_file(
        self,
        scan_run_id: int,
        file_path: str,
        practice: str | None = None,
    ) -> int:
        """
        Remove findings for a specific file in a scan run. If `practice` is
        given, only that (file, practice) pair is deleted; otherwise all
        findings for the file across all practices are removed.

        Returns the number of rows deleted.

        Used by the UI when the user clicks "Re-analyze this file" — we
        delete the old finding before inserting the new one so the
        per-practice counts stay correct and the validation log doesn't
        flag two findings for the same (file, practice) pair.
        """
        with self._conn() as conn:
            if practice is None:
                cur = conn.execute(
                    "DELETE FROM findings WHERE scan_run_id=? AND file_path=?",
                    (scan_run_id, file_path),
                )
            else:
                cur = conn.execute(
                    "DELETE FROM findings WHERE scan_run_id=? "
                    "AND file_path=? AND practice=?",
                    (scan_run_id, file_path, practice),
                )
            return cur.rowcount

    def delete_all_findings(self, scan_run_id: int) -> int:
        """
        Remove all findings for a scan run. Used when re-running the
        detection stage from scratch via the UI's "Rerun detection" button.
        Returns the number of rows deleted.
        """
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM findings WHERE scan_run_id=?",
                (scan_run_id,),
            )
            return cur.rowcount

    # ─── agent_registry ────────────────────────────────────────────────
    def register_agent(self, agent_id: str, agent_type: str) -> None:
        now = _now()
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO agent_registry
                   (agent_id, agent_type, status, started_at, last_heartbeat)
                   VALUES (?, ?, 'idle', ?, ?)
                   ON CONFLICT(agent_id) DO UPDATE
                   SET status='idle', started_at=?, last_heartbeat=?""",
                (agent_id, agent_type, now, now, now, now),
            )

    def update_agent_status(
        self, agent_id: str, status: str, current_task: str | None = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """UPDATE agent_registry
                   SET status=?, current_task=?, last_heartbeat=?
                   WHERE agent_id=?""",
                (status, current_task, _now(), agent_id),
            )

    def get_all_agents(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM agent_registry").fetchall()
        return [dict(r) for r in rows]

    def get_running_agents(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM agent_registry WHERE status='running'",
            ).fetchall()
        return [dict(r) for r in rows]

    # ─── stage_status ──────────────────────────────────────────────────
    def set_stage_state(
        self,
        scan_run_id: int,
        stage: str,
        state: str,
        progress: Dict[str, Any] | None = None,
        message: str | None = None,
    ) -> None:
        """
        Upsert stage state. Use the canonical STAGE_* and STATE_* constants.
        started_at is written only the first time the stage transitions
        to 'running' and then preserved.
        """
        if state not in VALID_STATES:
            raise ValueError(f"Invalid stage state: {state!r}")
        now = _now()
        started_at = now if state == STATE_RUNNING else None
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT started_at FROM stage_status WHERE scan_run_id=? AND stage=?",
                (scan_run_id, stage),
            ).fetchone()
            if existing and existing["started_at"] and state != STATE_RUNNING:
                started_at = existing["started_at"]

            conn.execute(
                """INSERT INTO stage_status
                   (scan_run_id, stage, state, progress, message, started_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(scan_run_id, stage) DO UPDATE
                   SET state=excluded.state,
                       progress=excluded.progress,
                       message=excluded.message,
                       started_at=COALESCE(stage_status.started_at, excluded.started_at),
                       updated_at=excluded.updated_at""",
                (scan_run_id, stage, state,
                 json.dumps(progress) if progress is not None else None,
                 message, started_at, now),
            )

    def get_stage_status(
        self, scan_run_id: int, stage: str | None = None,
    ) -> List[Dict[str, Any]] | Dict[str, Any] | None:
        """
        If `stage` is given, return a single dict (or None);
        otherwise return a list of all stages for the run.
        """
        with self._conn() as conn:
            if stage:
                row = conn.execute(
                    "SELECT * FROM stage_status WHERE scan_run_id=? AND stage=?",
                    (scan_run_id, stage),
                ).fetchone()
                if not row:
                    return None
                d = dict(row)
                d["progress"] = json.loads(d["progress"]) if d["progress"] else None
                return d
            rows = conn.execute(
                "SELECT * FROM stage_status WHERE scan_run_id=? ORDER BY id",
                (scan_run_id,),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["progress"] = json.loads(d["progress"]) if d["progress"] else None
            out.append(d)
        return out

    # ─── checkpoints ───────────────────────────────────────────────────
    def save_checkpoint(
        self, scan_run_id: int, stage: str, payload: Dict[str, Any],
    ) -> int:
        """
        Persist a named snapshot for a (scan_run, stage) pair.
        Overwrites any existing checkpoint for that pair.
        """
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO checkpoints (scan_run_id, stage, payload, created_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(scan_run_id, stage) DO UPDATE
                   SET payload=excluded.payload, created_at=excluded.created_at""",
                (scan_run_id, stage, json.dumps(payload), _now()),
            )
            return cur.lastrowid

    def load_checkpoint(
        self, scan_run_id: int, stage: str,
    ) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM checkpoints WHERE scan_run_id=? AND stage=?",
                (scan_run_id, stage),
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["payload"] = json.loads(d["payload"])
        return d

    def list_checkpoints(self, scan_run_id: int) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, scan_run_id, stage, created_at FROM checkpoints "
                "WHERE scan_run_id=? ORDER BY created_at",
                (scan_run_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ─── validation_log ────────────────────────────────────────────────
    def log_validation(
        self,
        scan_run_id: int,
        stage: str,
        check_name: str,
        passed: bool,
        details: str | None = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO validation_log
                   (scan_run_id, stage, check_name, passed, details, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (scan_run_id, stage, check_name, int(passed), details, _now()),
            )

    def get_validation_log(
        self, scan_run_id: int, stage: str | None = None,
    ) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            if stage:
                rows = conn.execute(
                    "SELECT * FROM validation_log WHERE scan_run_id=? AND stage=? ORDER BY id",
                    (scan_run_id, stage),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM validation_log WHERE scan_run_id=? ORDER BY id",
                    (scan_run_id,),
                ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["passed"] = bool(d["passed"])
            out.append(d)
        return out
