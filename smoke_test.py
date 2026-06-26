from __future__ import annotations
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from database.db import (
    AuditDatabase,
    STAGE_REPO_SCAN, STAGE_FILE_SELECTION, STAGE_VALIDATION,
    STAGE_DETECTION, STAGE_REPORT,
    STATE_COMPLETE, STATE_PARTIAL,
)
from src.llm import OllamaClient, Conversation
from agents.orchestrator_agent import OrchestratorAgent
from detectors.sim_async_parallel_conversation import (
    create_python_rl_sim_async_parallel_conversation,
)
from detectors.real_world_shadow_conversation import (
    create_python_rl_real_world_shadow_conversation,
)

DETECTORS = {
    "Simulation-Based (Async/Parallel)": create_python_rl_sim_async_parallel_conversation,
    "Real-World (Shadow Mode)":          create_python_rl_real_world_shadow_conversation,
}

# Filled in by build_sample_repo() at the start of main(), so the test is
# self-contained and needs no pre-existing fixture on disk.
SAMPLE_REPO = ""


def build_sample_repo(path: str) -> None:
    """
    Create a tiny throwaway repository the smoke test scans.

    Contains:
      * parallel_rollouts.py  -> strong async/parallel RL signal (selected)
      * ppo_parallel.yaml     -> parallel config (selected)
      * test_workers.py       -> a test file with no RL signal (not selected)
      * helper_notes.py       -> unrelated helper (not selected)
    """
    root = Path(path)
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    (root / "parallel_rollouts.py").write_text(
        "import multiprocessing as mp\n"
        "import gym\n"
        "\n"
        "def rollout_worker(env_name, queue):\n"
        "    env = gym.make(env_name)\n"
        "    obs = env.reset()\n"
        "    done = False\n"
        "    while not done:\n"
        "        action = env.action_space.sample()\n"
        "        obs, reward, done, info = env.step(action)\n"
        "        queue.put((obs, action, reward))\n"
        "\n"
        "def main():\n"
        "    queue = mp.Queue()\n"
        "    workers = [mp.Process(target=rollout_worker, args=('CartPole-v1', queue))\n"
        "               for _ in range(4)]\n"
        "    for w in workers:\n"
        "        w.start()\n"
        "    for w in workers:\n"
        "        w.join()\n"
    )

    (root / "ppo_parallel.yaml").write_text(
        "algorithm: PPO\n"
        "num_workers: 8\n"
        "num_envs_per_worker: 4\n"
        "rollout_fragment_length: 200\n"
        "train_batch_size: 4000\n"
        "framework: torch\n"
    )

    (root / "test_workers.py").write_text(
        "# A test file with no RL data-collection signal.\n"
        "def test_dummy():\n"
        "    assert 1 + 1 == 2\n"
    )

    (root / "helper_notes.py").write_text(
        "# An unrelated helper with no RL signal.\n"
        "def add(a, b):\n"
        "    return a + b\n"
    )


class MockLLM(OllamaClient):
    """
    Deterministic LLM stub.

    * Files whose name matches "parallel_rollouts.py" → supported=True
      for Simulation-Based (Async/Parallel), supported=False for shadow.
    * YAML configs → supported=True for Simulation-Based (Async/Parallel).
    * Everything else → supported=False.

    The response is always valid JSON that exercises our parser.
    """

    def __init__(self):
        self.base_url = "mock://"
        self.model = "mock-llm"
        self.max_tokens = 600
        self.timeout = 0
        self.call_log = []

    def chat(self, conversation: Conversation) -> str:
        last_user = ""
        first_system = ""
        for m in conversation.messages:
            if m["role"] == "user":
                last_user = m["content"]      # last one wins → real query
            elif m["role"] == "system" and not first_system:
                first_system = m["content"]

        filename = ""
        for line in last_user.splitlines():
            if line.strip().startswith("# FILE:"):
                filename = line.split("# FILE:", 1)[1].strip()
                break

        is_shadow_practice = "Shadow Mode" in first_system
        self.call_log.append({"file": filename, "shadow": is_shadow_practice})

        base = os.path.basename(filename)
        supported = False
        if not is_shadow_practice:
            # Async/parallel detection
            if base in ("parallel_rollouts.py", "ppo_parallel.yaml"):
                supported = True

        if supported:
            payload = {
                "supported": True,
                "evidence": [
                    {
                        "line_number": 24,
                        "code_snippet": (
                            "workers = [mp.Process(target=rollout_worker, ...) "
                            "for _ in range(4)]"
                        ),
                        "explanation": (
                            "Four rollout workers run in separate processes "
                            "collecting transitions in parallel."
                        ),
                    }
                ],
                "assets": {"notes": "mocked LLM response"},
            }
        else:
            payload = {"supported": False, "evidence": [], "assets": {}}

        return json.dumps(payload)


def assert_eq(name, actual, expected):
    if actual != expected:
        raise AssertionError(f"{name}: expected {expected!r}, got {actual!r}")
    print(f"  ✓ {name} = {actual!r}")


def assert_true(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name} FAILED {detail}")
    print(f"  ✓ {name}")


def run_full_pipeline(tmp_dir: Path):
    print("\n[smoke] ── Test 1: full pipeline (fresh run) ──")
    db_path = tmp_dir / "run1.db"
    reports_dir = tmp_dir / "reports1"

    db = AuditDatabase(str(db_path))
    orch = OrchestratorAgent(
        db=db,
        llm_client=MockLLM(),
        detectors=DETECTORS,
        max_files=10,
        reports_dir=str(reports_dir),
        verbose=False,
    )
    result = orch.run(SAMPLE_REPO)

    scan_run_id = result["scan_run_id"]
    assert_eq("scan_run_id", scan_run_id, 1)

    # Stage status: every stage should be complete
    stages = {s["stage"]: s["state"] for s in result["stage_status"]}
    for stage in (STAGE_REPO_SCAN, STAGE_FILE_SELECTION,
                  STAGE_VALIDATION, STAGE_DETECTION, STAGE_REPORT):
        assert_eq(f"stage[{stage}]", stages[stage], STATE_COMPLETE)

    cand_files = [os.path.basename(p) for p in result["candidate_files"]]
    assert_true("parallel_rollouts.py selected", "parallel_rollouts.py" in cand_files)
    assert_true("ppo_parallel.yaml selected",    "ppo_parallel.yaml"    in cand_files)
    assert_true("test_workers.py NOT selected",  "test_workers.py"      not in cand_files)

    findings = db.get_findings(scan_run_id)
    para_findings = [
        f for f in findings
        if "parallel_rollouts.py" in f["file_path"]
        and "Async/Parallel" in f["practice"]
    ]
    assert_true(
        "parallel_rollouts.py has async-parallel finding",
        len(para_findings) == 1 and para_findings[0]["supported"] is True,
    )

    ev = para_findings[0]["evidence"]
    assert_true(
        "evidence is structured (dicts with line_number)",
        isinstance(ev, list) and len(ev) > 0
        and all(isinstance(e, dict) and "line_number" in e for e in ev),
    )

    log = db.get_validation_log(scan_run_id)
    result_checks = {r["check_name"]: r["passed"] for r in log
                     if r["stage"] == STAGE_DETECTION}
    for check in ("all_files_analysed",
                  "supported_has_evidence",
                  "not_supported_has_no_evidence"):
        assert_eq(f"validation_log[{check}]", result_checks.get(check), True)

    paths = result["report_paths"]
    assert_true("JSON report exists", "json" in paths and Path(paths["json"]).exists())
    assert_true("HTML report exists", "html" in paths and Path(paths["html"]).exists())


    html = Path(paths["html"]).read_text()
    assert_true("HTML has evidence-table", "evidence-table" in html)
    assert_true("HTML has line number cell", "td class='line'" in html or "td class=\"line\"" in html)

    return scan_run_id, db_path, reports_dir


def run_standalone_tools(tmp_dir: Path, scan_run_id: int, db_path: Path):
    print("\n[smoke] ── Test 2: standalone tool invocations ──")
    reports_dir = tmp_dir / "reports2"

    db = AuditDatabase(str(db_path))
    orch = OrchestratorAgent(
        db=db,
        llm_client=MockLLM(),
        detectors=DETECTORS,
        max_files=10,
        reports_dir=str(reports_dir),
        verbose=False,
    )

    out = orch.regenerate_report(scan_run_id)
    assert_true("regenerate_report wrote JSON",
                Path(out["report_paths"]["json"]).exists())

    out = orch.revalidate_results(scan_run_id)
    assert_eq("revalidate_results overall_passed", out["overall_passed"], True)

    tools = orch.dispatcher.list_all_tools()
    servers_seen = {t["server"] for t in tools}
    for expected in ("repository_scanner_server", "file_selection_server",
                     "validation_server", "detection_server",
                     "report_generator_server"):
        assert_true(f"tool catalog includes {expected}", expected in servers_seen)


    resp = orch.dispatcher.call(
        "detection_server", "get_detection_status",
        {"scan_run_id": scan_run_id},
    )
    status = resp.get_data()
    assert_eq("direct tool: detection status state", status["state"], STATE_COMPLETE)


def run_resume(tmp_dir: Path):
    print("\n[smoke] ── Test 3: resume from checkpoint ──")
    db_path = tmp_dir / "run3.db"
    reports_dir = tmp_dir / "reports3"

    db = AuditDatabase(str(db_path))
    orch = OrchestratorAgent(
        db=db, llm_client=MockLLM(), detectors=DETECTORS,
        max_files=10, reports_dir=str(reports_dir), verbose=False,
    )

    scan_run_id = db.create_scan_run(SAMPLE_REPO, "mock-llm")
    orch.file_selection_agent.scan_repository(
        type("C", (), {"scan_run_id": scan_run_id,
                       "repo_root": SAMPLE_REPO,
                       "model": "mock-llm",
                       "all_source_files": [],
                       "repo_scan_summary": {},
                       "note": lambda s, *_a, **_k: None})(),
    )

    db2_path = tmp_dir / "run3b.db"
    reports_dir2 = tmp_dir / "reports3b"
    db2 = AuditDatabase(str(db2_path))
    orch2 = OrchestratorAgent(
        db=db2, llm_client=MockLLM(), detectors=DETECTORS,
        max_files=10, reports_dir=str(reports_dir2), verbose=False,
    )
    first = orch2.run(SAMPLE_REPO)
    run_id = first["scan_run_id"]


    initial_findings_count = len(db2.get_findings(run_id))
    orch3 = OrchestratorAgent(
        db=AuditDatabase(str(db2_path)),
        llm_client=MockLLM(), detectors=DETECTORS,
        max_files=10, reports_dir=str(reports_dir2), verbose=False,
    )
    mock = orch3.llm if hasattr(orch3, "llm") else None
    result2 = orch3.run(SAMPLE_REPO, scan_run_id=run_id)
    assert_true("resume did not duplicate findings",
                len(db2.get_findings(run_id)) == initial_findings_count)
    # Stages still complete
    stages = {s["stage"]: s["state"] for s in result2["stage_status"]}
    assert_eq("resumed stage[detection]", stages[STAGE_DETECTION], STATE_COMPLETE)


def run_llm_contract_failure(tmp_dir: Path):
    print("\n[smoke] ── Test 4: LLM contract violation triggers parse_warning ──")

    class BadLLM(OllamaClient):
        def __init__(self):
            self.model = "bad-llm"
            self.base_url = "mock://"
            self.max_tokens = 600
            self.timeout = 0
        def chat(self, conversation):
            return "Here's my thoughts... definitely looks fishy."

    db_path = tmp_dir / "run4.db"
    reports_dir = tmp_dir / "reports4"
    db = AuditDatabase(str(db_path))
    orch = OrchestratorAgent(
        db=db, llm_client=BadLLM(), detectors=DETECTORS,
        max_files=10, reports_dir=str(reports_dir), verbose=False,
    )
    result = orch.run(SAMPLE_REPO)
    findings = db.get_findings(result["scan_run_id"])
    warnings = [f for f in findings if f.get("parse_warning")]
    assert_true("every finding has parse_warning",
                len(warnings) == len(findings) and len(findings) > 0)
    # Detection state should be PARTIAL because warnings > 0
    stages = {s["stage"]: s["state"] for s in result["stage_status"]}
    assert_eq("detection state w/ warnings", stages[STAGE_DETECTION], STATE_PARTIAL)


def run_selection_cap(tmp_dir: Path):
    print("\n[smoke] ── Test 5: selection cap triggers 'partial' state ──")
    db_path = tmp_dir / "run5.db"
    reports_dir = tmp_dir / "reports5"
    db = AuditDatabase(str(db_path))
    orch = OrchestratorAgent(
        db=db, llm_client=MockLLM(), detectors=DETECTORS,
        max_files=1,                # force truncation — we have 2 good candidates
        reports_dir=str(reports_dir), verbose=False,
    )
    result = orch.run(SAMPLE_REPO)
    stages = {s["stage"]: s["state"] for s in result["stage_status"]}
    assert_eq("selection state w/ cap",
              stages[STAGE_FILE_SELECTION], STATE_PARTIAL)


def main():
    global SAMPLE_REPO
    tmp_dir = Path(tempfile.mkdtemp(prefix="rl_auditor_smoke_"))
    SAMPLE_REPO = str(tmp_dir / "sample_rl_repo")
    build_sample_repo(SAMPLE_REPO)
    print(f"[smoke] Using tmp dir: {tmp_dir}")
    print(f"[smoke] Built sample repo at: {SAMPLE_REPO}")
    try:
        scan_run_id, db_path, _ = run_full_pipeline(tmp_dir)
        run_standalone_tools(tmp_dir, scan_run_id, db_path)
        run_resume(tmp_dir)
        run_llm_contract_failure(tmp_dir)
        run_selection_cap(tmp_dir)
        print("\n[smoke] ALL TESTS PASSED ✓")
    except AssertionError as exc:
        print(f"\n[smoke] TEST FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()