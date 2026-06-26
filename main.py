"""
main.py
-------
CLI entry point for the RL Auditor.

Usage
-----
Full scan (new run):
    python main.py /path/to/repo

Resume an existing run from the DB (skips stages already marked 'complete'):
    python main.py /path/to/repo --scan-run-id 5

Standalone tool invocations (bypass the full pipeline):
    python main.py --regenerate-report 5
    python main.py --rerun-detection 5
    python main.py --list-tools              # dump the MCP tool catalog

Any of these can be combined with --db-path / --model / --reports-dir etc.
"""

import argparse
import json
import sys

import config
from database.db import AuditDatabase
from src.llm import OllamaClient
from agents.orchestrator_agent import OrchestratorAgent

# ── Detector registry ─────────────────────────────────────────────────────
# Add new practices here — nothing else needs to change.
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

DETECTORS = {
    "Simulation-Based (Async/Parallel)": create_python_rl_sim_async_parallel_conversation,
    "Real-World (Shadow Mode)":          create_python_rl_real_world_shadow_conversation,
    "Hybrid (Sim-to-Real)":              create_python_rl_hybrid_sim_to_real_conversation,
    "Offline (Batch)":                   create_python_rl_offline_batch_conversation,
    "Human-in-the-Loop":                 create_python_rl_human_in_the_loop_conversation,
    "League-Based Curriculum":           create_python_rl_league_based_conversation,
    "Preference-Data Collection (Pairwise Comparison)": create_python_rl_preference_based_conversation,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Agentic LLM-based RL data collection practice auditor.",
    )
    p.add_argument("repo_path", nargs="?",
                   help="Path to the local repository to scan.")
    p.add_argument("--model",       default=config.OLLAMA_MODEL)
    p.add_argument("--base-url",    default=config.OLLAMA_BASE_URL)
    p.add_argument("--max-files",   type=int, default=config.MAX_FILES)
    p.add_argument("--reports-dir", default=config.REPORTS_DIR)
    p.add_argument("--db-path",     default=str(config.DB_PATH))
    p.add_argument("--quiet",       action="store_true")
    p.add_argument("--practices",   default=None,
                   help="Comma-separated substrings; only practices whose name "
                        "contains one of them are run (e.g. --practices offline). "
                        "Default: run all practices.")

    # Resume / standalone modes
    p.add_argument("--scan-run-id",          type=int, default=None,
                   help="Resume an existing run by ID.")
    p.add_argument("--regenerate-report",    type=int, default=None,
                   help="Generate a report for an existing scan_run_id and exit.")
    p.add_argument("--rerun-detection",      type=int, default=None,
                   help="Re-run detection for an existing scan_run_id and exit.")
    p.add_argument("--revalidate-results",   type=int, default=None,
                   help="Re-run post-stage result validation and exit.")
    p.add_argument("--list-tools",           action="store_true",
                   help="Print the MCP tool catalog as JSON and exit.")
    p.add_argument("--stage-status",         type=int, default=None,
                   help="Print stage status for scan_run_id and exit.")
    return p.parse_args()


def build_orchestrator(args: argparse.Namespace) -> OrchestratorAgent:
    db = AuditDatabase(args.db_path)
    llm = OllamaClient(
        base_url=args.base_url,
        model=args.model,
        max_tokens=config.OLLAMA_MAX_TOKENS,
        timeout=config.OLLAMA_TIMEOUT,
    )
    detectors = DETECTORS
    if getattr(args, "practices", None):
        wanted = [w.strip().lower() for w in args.practices.split(",") if w.strip()]
        detectors = {name: fn for name, fn in DETECTORS.items()
                     if any(w in name.lower() for w in wanted)}
        if not detectors:
            raise SystemExit(
                f"[main] No practices matched {args.practices!r}. "
                f"Available: {list(DETECTORS)}"
            )
        print(f"[main] Running ONLY: {list(detectors)}")
    return OrchestratorAgent(
        db=db,
        llm_client=llm,
        detectors=detectors,
        max_files=args.max_files,
        reports_dir=args.reports_dir,
        verbose=not args.quiet,
    )


def print_detection_summary(scan_result: dict) -> None:
    print("\n[main] ── Stage Status ──────────────────────────────")
    for s in scan_result.get("stage_status", []) or []:
        print(f"  {s['stage']:<22} {s['state']:<14} {s.get('message') or ''}")

    print("\n[main] ── Agent Status ───────────────────────────────")
    for agent_id, status in scan_result.get("agent_summary", {}).items():
        icon = "✓" if status == "complete" else "✗"
        print(f"  {icon}  {agent_id:<30} {status}")

    print("\n[main] ── Detection Results ──────────────────────────")
    all_ok = True
    for practice, data in scan_result["results_by_practice"].items():
        ok = data["summary"]["compliance_detected"]
        icon = "✓" if ok else "✗"
        print(f"  {icon}  {practice}")
        if not ok:
            all_ok = False

    log = scan_result.get("validation_log") or []
    if log:
        print("\n[main] ── Validation Log ─────────────────────────────")
        for row in log:
            icon = "✓" if row["passed"] else "✗"
            print(f"  {icon}  [{row['stage']}] {row['check_name']}: {row.get('details') or ''}")

    paths = scan_result.get("report_paths") or {}
    if paths:
        print("\n[main] Reports:")
        for fmt, pth in paths.items():
            print(f"  {fmt.upper()}: {pth}")


def main() -> None:
    args = parse_args()

    # ── Standalone / introspection modes first ──
    if args.list_tools:
        orch = build_orchestrator(args)
        print(json.dumps(orch.describe_all(), indent=2))
        sys.exit(0)

    if args.stage_status is not None:
        orch = build_orchestrator(args)
        rows = orch.db.get_stage_status(args.stage_status)
        print(json.dumps(rows, indent=2, default=str))
        sys.exit(0)

    if args.regenerate_report is not None:
        orch = build_orchestrator(args)
        out = orch.regenerate_report(args.regenerate_report)
        print(json.dumps(out, indent=2))
        sys.exit(0)

    if args.rerun_detection is not None:
        orch = build_orchestrator(args)
        out = orch.rerun_detection(args.rerun_detection)
        print(f"[main] Rerun complete: "
              f"{out.get('total_pairs', 0)} pairs, "
              f"{out.get('warnings', 0)} warnings")
        sys.exit(0)

    if args.revalidate_results is not None:
        orch = build_orchestrator(args)
        out = orch.revalidate_results(args.revalidate_results)
        print(json.dumps(out, indent=2))
        sys.exit(0 if out.get("overall_passed") else 1)

    # ── Full pipeline (new run or resume) ──
    if not args.repo_path:
        print("[main] repo_path is required (or use one of the --standalone flags).",
              file=sys.stderr)
        sys.exit(2)

    print(f"[main] Model        : {args.model}")
    print(f"[main] Ollama URL   : {args.base_url}")
    print(f"[main] Max files    : {args.max_files}")
    print(f"[main] Database     : {args.db_path}")
    print(f"[main] Practices    : {list(DETECTORS.keys())}")

    orch = build_orchestrator(args)
    try:
        scan_result = orch.run(
            repo_root=args.repo_path,
            scan_run_id=args.scan_run_id,
        )
    except ValueError as exc:
        print(f"[main] Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except ConnectionError as exc:
        print(f"[main] LLM connection error: {exc}", file=sys.stderr)
        sys.exit(2)
    except RuntimeError as exc:
        print(f"[main] Pipeline error: {exc}", file=sys.stderr)
        sys.exit(1)

    print_detection_summary(scan_result)

    # Exit code: 0 if any detection compliance is true, 3 otherwise.
    all_ok = all(
        data["summary"]["compliance_detected"]
        for data in scan_result["results_by_practice"].values()
    )
    sys.exit(0 if all_ok else 3)


if __name__ == "__main__":
    main()