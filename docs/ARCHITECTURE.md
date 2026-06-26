# RL Auditor v3 — Architecture & Usage

This document is the canonical reference for how the RL Auditor is
assembled: the MCP servers, the agents that drive them, the stage/state
model, the two-tier memory system, and how external systems invoke
specific tools.

To run the system in a container (including opening the dashboard with the
bundled scan results, no Ollama required), see **`README_DOCKER.md`**.

---

## 1. Big picture

```
                         OrchestratorAgent
                                 │
                   ┌─────────────┼─────────────────────────┐
                   │             │                         │
          AuditDatabase     MCPDispatcher            Multi-agent system
        (long-term memory)       │                   (stage owners)
                                 │
   ┌──────────────────┬──────────┼────────────────┬─────────────────────┐
   │                  │          │                │                     │
repository_     file_selection  validation   detection             report_generator
scanner_server  _server         _server      _server               _server
   │                  │          │                │                     │
   └── stage 1 ───────┴── stage 2 ─ stage 3 ─────── stage 4 ──── stage 5/6 ──┘
```

Key design choices:

* **MCP is the only wire between agents and work.** Agents don't call
  internal helpers directly; they go through `MCPDispatcher.call(server,
  tool, args)`. This means *any* tool can be invoked from the outside
  with the same call shape — no privileged internal path.
* **Five thin servers instead of one fat one.** Each server owns exactly
  one pipeline stage's work. The split lets an external system call just
  `report_generator_server.generate_report(scan_run_id=5)` without ever
  touching selection or detection.
* **Two-tier memory.** Short-term = `ScanContext` (in-process dataclass
  passed between tools in one run). Long-term = SQLite. Every write to
  short-term is mirrored to long-term, so any scan is resumable and any
  stage is re-invokable.
* **Stage status is first-class.** Every (scan_run, stage) pair has one
  row in `stage_status` that records `not_started / running / partial /
  complete / failed` plus a JSON `progress` blob. That row is the source
  of truth callers should query to know what's been done.

---

## 2. The MCP servers and their tools

Every tool below is reachable via the dispatcher using
`(server_name, tool_name, arguments_dict)`. A flat tool catalog is also
available at `main.py --list-tools`.

### 2.1 `repository_scanner_server`

Purpose: walk a directory tree, classify files by extension, persist an
inventory checkpoint. Does **not** filter or rank — that's the next
server's job.

| Tool | Description | Required args |
|---|---|---|
| `scan_repository` | Walk `repo_root`, collect source files, save a checkpoint, update `stage_status(repository_scan)` | `repo_root`, `scan_run_id`, optional `extensions` |
| `get_repo_scan_summary` | Return the checkpoint payload from a prior scan | `scan_run_id` |

### 2.2 `file_selection_server`

Purpose: apply keyword pre-filtering + ranking to the repo-scan
inventory and persist candidates in `candidate_files`.

| Tool | Description | Required args |
|---|---|---|
| `select_files` | Filter + rank + cap at `max_files`. State is `partial` if the cap truncated the list, otherwise `complete`. | `scan_run_id`, optional `max_files`, `repo_root` (fallback) |
| `list_candidates` | Inspect candidates for a scan run, optionally filtered by status | `scan_run_id`, optional `status` |
| `get_selection_status` | Return the `stage_status` row for `file_selection` | `scan_run_id` |

### 2.3 `validation_server`

Purpose: deterministic rule-based validation. Two kinds:

(A) Pre-detection file checks (existence, extension, size, readable).
(B) Post-stage result sanity checks (see §4).

**No LLM is ever involved in this server.**

| Tool | Description | Required args |
|---|---|---|
| `validate_candidates` | File-level checks; marks each file `validated` or `rejected` | `scan_run_id` |
| `validate_detection_results` | Runs the three post-stage result checks; logs each to `validation_log` | `scan_run_id` |
| `get_validation_report` | Consolidated summary: validated/rejected counts + the validation log | `scan_run_id` |

### 2.4 `detection_server`

Purpose: run LLM-based practice detection and persist structured
findings. Writes evidence as JSON objects with `line_number`,
`code_snippet`, `explanation`. The LLM may also attach free-form
`assets` which are stored verbatim.

| Tool | Description | Required args |
|---|---|---|
| `run_detection` | Run every registered practice against every validated file | `scan_run_id`, optional `practices` |
| `run_detection_on_files` | Run detection against a specific subset of files | `scan_run_id`, `file_paths`, optional `practices` |
| `get_findings` | Return all stored findings for a scan run, grouped by practice | `scan_run_id` |
| `get_detection_status` | Return the `stage_status` row for `detection` | `scan_run_id` |

### 2.5 `report_generator_server`

Purpose: build JSON and HTML reports from the findings already in the
DB. Runs fully offline — useful for re-generating a report after edits
to the report template, or for callers that only want the output.

| Tool | Description | Required args |
|---|---|---|
| `generate_report` | Write JSON + HTML reports to disk | `scan_run_id`, optional `format` (`json`\|`html`\|`all`), `stem` |
| `get_report_paths` | Return the last-written report paths for a scan run | `scan_run_id` |
| `get_report_status` | Return the `stage_status` row for `report_generation` | `scan_run_id` |

### 2.6 Adding a new server

1. Create `mcp_servers/<name>_server.py` with a class inheriting
   `BaseMCPServer`.
2. Decorate handler methods with `@register_tool(name, description,
   input_schema)`.
3. Register it in `OrchestratorAgent.__init__`:
   `dispatcher.register_server(YourServer(db))`.

That's it — no other wiring required. A smoke test for the new tools is
`main.py --list-tools | jq '.[]|.tools[].name'`.

---

## 3. The multi-agent layer

One agent per stage. Each agent is a thin wrapper around the dispatcher
that (a) registers itself in `agent_registry`, (b) updates its own
status as tasks progress, and (c) owns the ScanContext writes for its
stage.

| Agent | Owns | Produces |
|---|---|---|
| `FileSelectionAgent` | repository_scanner_server, file_selection_server | `ctx.all_source_files`, `ctx.selected_files` |
| `ValidationAgent` | validation_server (both sides) | `ctx.validated_files`, `ctx.rejected_files`, `validation_log` entries |
| `DetectionAgent` | detection_server | `ctx.findings` |
| `ReportAgent` | report_generator_server | `ctx.report_paths` |
| `OrchestratorAgent` | all of the above | coordinates stages, handles resume logic |

**Why the split matters.** Future agents can plug in without touching
anyone else:

* `SuggestionAgent` — for each non-supported finding, ask the LLM for a
  code-level fix
* `TestGenerationAgent` — generate unit tests targeting detected
  practices
* `CodeGenerationAgent` — scaffold compliant code from an empty
  repository

Each of those would own its own MCP server and its own slot in
`agent_registry`.

---

## 4. Stage and state model

Every scan run has these five stages, in order:

```
repository_scan → file_selection → validation → detection → report_generation
```

Every stage is in exactly one of these states at any time:

| State | Meaning |
|---|---|
| `not_started` | seeded on scan-run creation; no work has begun |
| `running` | the stage is currently executing |
| `partial` | some work done but not all (e.g., detection ran but some files produced parse warnings; selection capped by `max_files`) |
| `complete` | done, nothing else to do |
| `failed` | hit an unrecoverable error |

Every stage also carries a JSON `progress` blob (e.g.
`{"processed": 12, "total": 25, "warnings": 2}`) and a human-readable
`message`.

### 4.1 Querying state

```python
# Full state of a run
db.get_stage_status(scan_run_id)

# One specific stage
db.get_stage_status(scan_run_id, STAGE_DETECTION)
```

From the CLI:

```bash
python main.py --stage-status 5
```

### 4.2 Validation checks

After detection, the validation agent runs three deterministic checks
against the stored findings. They are logged to `validation_log` and
also returned as a single JSON report.

| Check | Pass condition |
|---|---|
| `all_files_analysed` | every validated file has at least one finding per registered practice |
| `supported_has_evidence` | every `supported=True` finding has ≥1 evidence item |
| `not_supported_has_no_evidence` | every `supported=False` finding has empty evidence |

These are deliberately simple and deterministic. No LLM involvement —
that would be circular.

---

## 5. Memory

### 5.1 Long-term: SQLite (`rl_auditor.db`)

Tables:

| Table | Role |
|---|---|
| `scan_runs` | one row per scan; holds repo_root, model, overall status |
| `candidate_files` | per-file: status (`selected`/`validated`/`rejected`/`complete`), keyword_score, rejection_reason |
| `findings` | per (file, practice): `supported` bool, `evidence` JSON, `assets` JSON, `raw_response`, `parse_warning` |
| `agent_registry` | which agents are registered and their current status/heartbeat |
| `stage_status` | one row per (scan_run, stage) — see §4 |
| `checkpoints` | named snapshots (one per stage per run) for resume-from-checkpoint |
| `validation_log` | every validation check run: `passed`, `check_name`, `details` |

### 5.2 Short-term: `ScanContext`

Defined in `memory/short_term.py`. A dataclass passed between tool
invocations in one run. Fields:

* `scan_run_id`, `repo_root`, `model`
* `all_source_files`, `repo_scan_summary`
* `selected_files`, `selection_summary`
* `validated_files`, `rejected_files`
* `findings`
* `report_paths`
* `notes` (free-form breadcrumbs)

Contract with the DB:

1. Every write to `ScanContext` must also be written to the DB by the
   same tool that made the write.
2. Read access goes through `ensure_selected(db)`, `ensure_validated(db)`,
   `ensure_findings(db)` helpers — these repopulate from the DB if the
   field is empty, which is exactly how a standalone tool invocation
   (e.g. `--regenerate-report`) recovers upstream state.

### 5.3 Checkpoints

A checkpoint is a (scan_run_id, stage) → JSON payload written to the
`checkpoints` table. It captures the stage's outputs in a form the next
stage can consume without re-running the DB queries.

Who writes checkpoints:

| Stage | Checkpoint payload |
|---|---|
| `repository_scan` | `{all_source_files, repo_scan_summary}` |
| `file_selection` | `{selected_files, selection_summary}` |
| `validation` | `{validated_files, rejected_files}` |
| `detection` | `{files_analyzed, practices_run, warnings}` |
| `report_generation` | `{report_paths}` |

### 5.4 Resume logic

`OrchestratorAgent.run(repo_root, scan_run_id=X)` with an existing ID:

1. Hydrates a `ScanContext` from the DB + checkpoints.
2. For each stage, checks `stage_status` — skips any that are already
   `complete`.
3. Picks up from the earliest non-complete stage.

This is the hook point for future LangGraph migration — today we ship a
straight-line skip-if-complete policy; tomorrow the same logic slots
into a graph node's conditional edge.

---

## 6. The LLM contract (v3)

Prompts under `prompts/<practice>/` now ask the LLM to emit a single
JSON object:

```json
{
  "supported": true,
  "evidence": [
    {
      "line_number": 42,
      "code_snippet": "...",
      "explanation": "..."
    }
  ],
  "assets": {}
}
```

Rules enforced by the parser in `src/output_parser.py`:

1. If `supported=false`, `evidence` must be empty.
2. If `supported=true`, `evidence` must have ≥1 item.
3. Every evidence item has `line_number`, `code_snippet`, `explanation`.
4. `assets` is free-form and passed through verbatim for display.

To make the contract robust on a small local model, the detection client
also sends a JSON schema as the Ollama `format`, so the model is
constrained to the expected structure rather than free text.

**Fallbacks the parser tolerates** (each logged as `parse_warning` so
the validation stage sees them):

* JSON inside a ` ```json ` fence instead of raw JSON
* Legacy ` ```python ` code blocks (v2 format) — wrapped into the new
  shape
* The literal sentinel `"no evidence found"` at the start of the
  response
* Any unparseable reply falls back conservatively to
  `supported=false` with a `parse_warning`, rather than a fabricated
  positive.

---

## 7. Invoking tools externally

External systems (another Python app, a CI job, another MCP client) can
invoke any single tool without going through the orchestrator.

### 7.1 From Python, in-process

```python
from database.db import AuditDatabase
from src.llm import OllamaClient
from agents.orchestrator_agent import OrchestratorAgent
from main import DETECTORS

db  = AuditDatabase("rl_auditor.db")
llm = OllamaClient(model="qwen2.5-coder:7b")
orch = OrchestratorAgent(db=db, llm_client=llm, detectors=DETECTORS)

# Catalog:
tools = orch.dispatcher.list_all_tools()

# Just the report, against an existing scan:
orch.regenerate_report(scan_run_id=5)

# Just re-run detection for one practice:
orch.rerun_detection(scan_run_id=5, practices=["Real-World (Shadow Mode)"])

# Fine-grained: call any tool directly
resp = orch.dispatcher.call(
    "detection_server", "run_detection_on_files",
    {"scan_run_id": 5, "file_paths": ["/path/to/a.py"]},
)
print(resp.get_data())
```

### 7.2 From the CLI

```bash
# New scan
python main.py /path/to/repo

# Resume scan 5 (skips already-complete stages)
python main.py /path/to/repo --scan-run-id 5

# Standalone re-invocations against an existing scan
python main.py --regenerate-report    5
python main.py --rerun-detection      5
python main.py --revalidate-results   5
python main.py --stage-status         5
python main.py --list-tools
```

The same tools run unchanged inside the container; see
**`README_DOCKER.md`** for the Docker workflow.

---

## 8. Supported file types

`EXTENSION_TO_FILETYPE` in `src/file_filter.py` is the single source of
truth:

| Extension | File type label |
|---|---|
| `.py` | python |
| `.yaml`, `.yml` | yaml |
| `.java` | java |

Adding another language:

1. Add the extension → label entry to `EXTENSION_TO_FILETYPE`.
2. (Optional) Add language-specific keywords to `_PARALLEL_KEYWORDS` /
   `_ENV_HINTS` if the existing Python keywords are too narrow.
3. (Optional) Add a fence label in `detection_server._FENCE_LABEL` so
   the LLM sees the correct code fence.

---

## 9. Code layout

```
rl_auditor_v3/
├── agents/                          one agent per stage + orchestrator
│   ├── base_agent.py
│   ├── file_selection_agent.py
│   ├── validation_agent.py
│   ├── detection_agent.py
│   ├── report_agent.py
│   └── orchestrator_agent.py
├── database/
│   └── db.py                        long-term memory (SQLite)
├── detectors/                       few-shot conversation builders
│   ├── sim_async_parallel_conversation.py
│   ├── real_world_shadow_conversation.py
│   ├── hybrid_sim_to_real_conversation.py
│   ├── offline_batch_conversation.py
│   ├── human_in_the_loop_conversation.py
│   ├── league_based_conversation.py
│   └── preference_based_conversation.py
├── docs/
│   └── ARCHITECTURE.md              this file
├── mcp_servers/                     one server per pipeline stage
│   ├── base_server.py
│   ├── repository_scanner_server.py
│   ├── file_selection_server.py
│   ├── validation_server.py
│   ├── detection_server.py
│   └── report_generator_server.py
├── memory/
│   └── short_term.py                ScanContext dataclass
├── prompts/                         JSON-format prompts per practice
│   ├── sim_async_parallel/
│   ├── real_world_shadow/
│   ├── hybrid_sim_to_real/
│   ├── offline_batch/
│   ├── human_in_the_loop/
│   ├── league_based/
│   └── preference_based/
├── reports/                         generated JSON + HTML go here
├── src/
│   ├── file_filter.py
│   ├── llm.py
│   ├── output_parser.py
│   └── report_generator.py
├── config.py
└── main.py
```