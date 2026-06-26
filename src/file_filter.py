from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

EXTENSION_TO_FILETYPE: Dict[str, str] = {
    ".py":   "python",
    ".yaml": "yaml",
    ".yml":  "yaml",
    ".java": "java",
}

_SOURCE_EXTENSIONS = set(EXTENSION_TO_FILETYPE.keys())


def classify_file_type(path: str) -> str:
    return EXTENSION_TO_FILETYPE.get(Path(path).suffix.lower(), "unknown")


_LIBRARY_TOKENS: List[str] = [
    # generic RL frameworks (any practice)
    "ray", "rllib", "gym", "gymnasium", "stable_baselines3", "sb3_contrib",
    "tianshou", "acme", "torchrl", "rl4j", "dopamine", "cleanrl", "agilerl",
    # simulation / async-parallel & sim-to-real
    "mujoco", "mujoco_py", "dm_control", "isaacgym", "isaacgymenvs",
    "omni", "brax", "envpool", "pybullet",
    # multi-agent / league / self-play
    "pettingzoo", "open_spiel", "openspiel", "pyspiel", "melting_pot",
    "supersuit",
    # offline / batch
    "d4rl", "minari", "rl_unplugged", "d3rlpy",
    # human-in-the-loop / demonstrations
    "imitation", "robomimic", "robosuite",
    # preference / RLHF
    "trl", "trlx", "openrlhf", "rlhf",
]

_IDENTIFIER_TOKENS: List[str] = [
    # ── Simulation-Based (Async/Parallel) ──
    "num_rollout_workers", "num_envs_per_worker", "remote_worker_envs",
    "num_cpus_per_worker", "num_gpus_per_worker", "subprocvecenv",
    "asyncvectorenv", "syncvectorenv", "vectorenv", "vec_env", "make_vec_env",
    "rollout_worker", "rollout_fragment_length", "rollout_workers",
    "num_workers", "num_envs", "envrunner", "env_runner", "sample_collector",
    "replay_buffer", "rollout", "collector", "sampler", "actor", "learner",
    "worker_id", "parallel_envs", "executorservice", "forkjoinpool",
    "processpoolexecutor", "threadpoolexecutor", "multiprocessing",
    "ray_remote",
    # ── Real-World (Shadow Mode) ──
    "shadow_mode", "shadow", "shadow_policy", "intervention",
    "human_intervention", "takeover", "safety_driver", "safety_monitor",
    "disengagement", "hypothetical_action", "logged_action", "counterfactual",
    "production_traffic", "live_traffic", "telemetry", "fleet",
    # ── Hybrid (Sim-to-Real / Domain Randomization) ──
    "domain_randomization", "dynamics_randomization", "physics_randomization",
    "randomization", "randomize_dynamics", "randomize_friction",
    "randomize_mass", "randomize_physics", "randomization_range",
    "domain_rand", "sim_to_real", "sim2real", "real_world_finetune",
    # ── Offline (Batch / Historical) ──
    "offline", "offline_rl", "offline_dataset", "replay_dataset",
    "static_dataset", "fixed_dataset", "behavior_policy", "behaviour_policy",
    "logged_dataset", "batch_rl", "trajectory_dataset", "qlearning_dataset",
    "conservative_q", "cql", "iql", "awac", "bcq", "brac", "td3_bc",
    # ── Human-in-the-Loop ──
    "human_in_the_loop", "human_feedback", "human_input", "ask_human",
    "query_user", "human_label", "human_reward", "oracle", "demonstration",
    "demonstrations", "expert_action", "expert_label", "expert_trajectory",
    "behavior_cloning", "behaviour_cloning", "dagger", "teleop",
    "teleoperation", "corrective_action",
    # ── League-Based Curriculum (Self-Play) ──
    "self_play", "selfplay", "league", "exploiter", "main_agent",
    "main_exploiter", "league_exploiter", "fictitious_self_play", "pfsp",
    "prioritized_fictitious", "nash", "opponent", "opponent_pool",
    "policy_pool", "past_versions", "snapshot", "payoff", "matchmaking",
    "win_rate",
    # ── Preference-Data Collection ──
    "preference", "preferences", "preference_pair", "preference_dataset",
    "preference_buffer", "preference_label", "pairwise", "comparison",
    "comparisons", "segment_pair", "chosen", "rejected", "reward_model",
    "reward_modeling", "bradley_terry", "ranking", "dueling",
]

_CALL_TOKENS: List[str] = [
    # sim / async-parallel
    "step_async", "step_wait", "remote", "make_vec_env",
    # offline
    "get_dataset", "qlearning_dataset", "load_dataset",
    # human-in-the-loop
    "query_human", "ask_human", "request_label", "get_human_feedback",
    # hybrid / sim-to-real
    "intervene", "randomize", "randomize_dynamics", "apply_randomization",
    # preference
    "query_human_preference", "query_preference", "collect_preferences",
    "add_comparison", "sample_segment_pair", "preference_loss",
    "compute_preference_loss",
]

_ENV_TOKENS: List[str] = [
    "env.step", "gym.make", "gymnasium.make", "env.reset", "envs.step",
    "vec_env.step", "mdp.step", "environment.step",
]

_STEM_TOKENS = {
    # offline / batch
    "offline", "terminals", "transition", "transitions", "trajectory",
    "trajectories", "historical", "logged", "behavior", "behaviour",
    "d4rl", "minari", "cql", "iql", "awac", "bcq", "brac",
    # human-in-the-loop / demonstrations
    "human", "oracle", "demonstration", "demonstrations", "teleop",
    "teleoperation", "expert", "dagger", "annotator", "feedback", "intervened",
    "intervention",
    # real-world shadow
    "shadow", "takeover", "disengagement", "counterfactual", "telemetry",
    "hypothetical",
    # hybrid / sim-to-real
    "randomization", "randomize", "sim2real",
    # league / self-play
    "selfplay", "league", "exploiter", "opponent", "nash", "payoff",
    "matchmaking", "pfsp", "fictitious",
    # preference
    "preference", "preferences", "pairwise", "comparison", "comparisons",
    "chosen", "rejected", "bradley", "dueling",
    # simulation / async-parallel (distinctive)
    "rollout", "subproc", "envrunner", "multiprocessing",
}

_NONALNUM_RE = re.compile(r"[^A-Za-z0-9]+")
_CAMEL_RE    = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _subtokens(text: str) -> set:
    """Lowercased sub-tokens of every identifier in ⁠ text ⁠."""
    out: set = set()
    for chunk in _NONALNUM_RE.split(text):
        if not chunk:
            continue
        for seg in _CAMEL_RE.split(chunk):
            if seg:
                out.add(seg.lower())
    return out


def _word_re(token: str) -> "re.Pattern[str]":
    """Word-boundary match for a (possibly dotted) identifier token."""
    return re.compile(
        r"(?<![A-Za-z0-9_])" + re.escape(token) + r"(?![A-Za-z0-9_])",
        re.IGNORECASE,
    )


def _call_re(name: str) -> "re.Pattern[str]":
    """Match an identifier used as a call: name( ."""
    return re.compile(
        r"(?<![A-Za-z0-9_])" + re.escape(name) + r"\s*\(",
        re.IGNORECASE,
    )


def _import_re(lib: str) -> "re.Pattern[str]":

    return re.compile(
        r"^\s*(?:import\s+" + re.escape(lib) + r"(?:\.\w+)*(?![A-Za-z0-9_])"
        r"|from\s+" + re.escape(lib) + r"(?:\.\w+)*\s+import)",
        re.IGNORECASE | re.MULTILINE,
    )


def _dotted_use_re(lib: str) -> "re.Pattern[str]":
    """Match dotted attribute use of ⁠ lib ⁠, e.g. rllib.algorithms."""
    return re.compile(
        r"(?<![A-Za-z0-9_])" + re.escape(lib) + r"\.\w+",
        re.IGNORECASE,
    )


# Pre-compile everything once at import time.
_LIBRARY_IMPORT_PATTERNS = [(lib, _import_re(lib), _dotted_use_re(lib))
                            for lib in _LIBRARY_TOKENS]
_IDENTIFIER_PATTERNS = [_word_re(t) for t in _IDENTIFIER_TOKENS]
_CALL_PATTERNS       = [_call_re(t) for t in _CALL_TOKENS]
_ENV_PATTERNS        = [_word_re(t) for t in _ENV_TOKENS]

# Flat lowercase list for the (generous) smart-extraction pass below.
_ALL_KEYWORDS: List[str] = sorted(set(
    [t.lower() for t in _LIBRARY_TOKENS]
    + [t.lower() for t in _IDENTIFIER_TOKENS]
    + [t.lower() for t in _CALL_TOKENS]
    + [t.lower() for t in _ENV_TOKENS]
    + list(_STEM_TOKENS)
))

_SKIP_DIRS = {
    ".git", ".venv", "venv", "_pycache_", "build", "dist", "node_modules",
    "target",  # Java/Maven build output
}

_SKIP_PATH_FRAGMENTS = {
    "/tests/", "/test_", "/benchmark",
    "/submit", "/resume", "/eval", "/evals/",
    "/docs/", "/examples/",
}

SMART_EXTRACT_THRESHOLD: int = 300
CONTEXT_LINES: int = 8


def extract_relevant_sections(
    code: str, context: int = CONTEXT_LINES,
) -> Tuple[str, bool]:
    """
    For large files: keep only lines containing relevant keywords, plus
    ⁠ context ⁠ lines around each, plus the top 20 lines (imports / class
    headers). Returns (extracted_code, was_extracted).
    """
    lines = code.splitlines()
    if len(lines) <= SMART_EXTRACT_THRESHOLD:
        return code, False

    relevant: set = set()
    for i, line in enumerate(lines):
        lower = line.lower()
        if any(kw in lower for kw in _ALL_KEYWORDS):
            start = max(0, i - context)
            end = min(len(lines), i + context + 1)
            for j in range(start, end):
                relevant.add(j)

    if not relevant:
        fallback = "\n".join(lines[:SMART_EXTRACT_THRESHOLD])
        fallback += (
            f"\n\n# [FALLBACK TRUNCATION: no keywords found, "
            f"showing first {SMART_EXTRACT_THRESHOLD} of {len(lines)} lines]"
        )
        return fallback, True

    for i in range(min(20, len(lines))):
        relevant.add(i)

    sorted_idx = sorted(relevant)
    out, prev = [], -1
    for idx in sorted_idx:
        if prev != -1 and idx > prev + 1:
            out.append(f"# ... [{idx - prev - 1} lines omitted] ...")
        out.append(lines[idx])
        prev = idx
    extracted = "\n".join(out)
    extracted += (
        f"\n\n# [SMART EXTRACTION: {len(out)} of {len(lines)} lines shown]"
    )
    return extracted, True


def collect_source_files(
    repo_root: str,
    extensions: set | None = None,
) -> List[str]:
    """
    Recursively walk repo_root and return all files whose extension is in
    ⁠ extensions ⁠ (default: EXTENSION_TO_FILETYPE keys).
    """
    exts = extensions or _SOURCE_EXTENSIONS
    results: List[str] = []
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for fname in files:
            if Path(fname).suffix.lower() in exts:
                results.append(os.path.join(root, fname))
    return results


def is_skippable_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    return any(frag in normalized for frag in _SKIP_PATH_FRAGMENTS)


def _scan_text(text: str) -> Tuple[int, int, int]:
    """
    Return (library_weight, distinct_signals, total_occurrences).

      library_weight    : 3 per actually-imported RL library, 1 per dotted use.
      distinct_signals  : number of DISTINCT evidence keys found — each full
                          identifier token, call token, env token, and stem
                          counts once. "Distinct" so that one keyword repeated
                          twice is still a single signal.
      total_occurrences : raw match count, used only for ranking.
    """
    library_weight = 0
    for _lib, import_re, dotted_re in _LIBRARY_IMPORT_PATTERNS:
        if import_re.search(text):
            library_weight += 3       # strong: a real import
        elif dotted_re.search(text):
            library_weight += 1       # medium: dotted attribute use

    signals: set = set()
    occurrences = 0

    for tok, pat in zip(_IDENTIFIER_TOKENS, _IDENTIFIER_PATTERNS):
        n = len(pat.findall(text))
        if n:
            signals.add("id:" + tok)
            occurrences += n
    for tok, pat in zip(_CALL_TOKENS, _CALL_PATTERNS):
        n = len(pat.findall(text))
        if n:
            signals.add("call:" + tok)
            occurrences += n
    for tok, pat in zip(_ENV_TOKENS, _ENV_PATTERNS):
        n = len(pat.findall(text))
        if n:
            signals.add("env:" + tok)
            occurrences += n

    subtoks = _subtokens(text)
    for stem in _STEM_TOKENS:
        if stem in subtoks:
            signals.add("stem:" + stem)
            occurrences += 1

    return library_weight, len(signals), occurrences


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None


def passes_keyword_filter(path: str) -> bool:
    """
    A file qualifies for LLM analysis when it shows genuine RL evidence:

      * it actually imports / uses an RL library (import-aware), OR
      * it has at least two DISTINCT identifier / call / env / stem signals.

    Matching is precise (word-boundary, import-aware, sub-token split on
    underscores and camelCase) so there are no "array"→"ray" false positives
    and no bare-token or comment-only matches, while covering every registered
    practice — so a relevant file is not silently dropped.
    """
    text = _read(path)
    if text is None:
        return False
    library_weight, distinct_signals, _ = _scan_text(text)
    return library_weight >= 1 or distinct_signals >= 2


def keyword_score(path: str) -> int:
    """Relevance score used only for ranking candidates (higher = first)."""
    text = _read(path)
    if text is None:
        return -1
    library_weight, _, occurrences = _scan_text(text)
    return library_weight + occurrences


def collect_python_files(repo_root: str) -> List[str]:
    return collect_source_files(repo_root, {".py"})