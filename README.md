# RL Auditor v3

Agentic LLM-based auditor that detects RL data-collection practices in a
source repository. v3 is a ground-up refactor with a multi-agent
architecture, a proper MCP tool boundary, a two-tier memory system, and
a structured LLM output contract.

## Quickstart

```bash
# New full scan
python main.py /path/to/repo

# Resume an existing scan (skips stages already marked 'complete')
python main.py /path/to/repo --scan-run-id 5

# Standalone re-invocations against an existing scan
python main.py --regenerate-report    5
python main.py --rerun-detection      5
python main.py --revalidate-results   5
python main.py --stage-status         5
python main.py --list-tools
```

## Requirements

* Python 3.10+
* `requests` (for the Ollama client)
* Ollama running locally (override via `--base-url` / `--model` or env vars)

## Smoke test (no Ollama needed)

```bash
python smoke_test.py
```

The smoke test uses a mocked LLM and verifies:

1. Full pipeline against a sample repo
2. Standalone tool invocations (regenerate_report, revalidate_results, direct dispatcher calls)
3. Resume from checkpoint (no duplicate findings)
4. LLM contract violations produce `parse_warning` and flip detection state to `partial`
5. Selection cap triggers `partial` state

## Documentation

Full architecture notes, tool catalog, stage/state/memory model, and
roadmap hooks (LangGraph migration, human-in-the-loop, LLM-driven file
selection) are in **`docs/ARCHITECTURE.md`**.
