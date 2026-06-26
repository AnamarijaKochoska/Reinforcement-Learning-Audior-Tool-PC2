# Running the RL Data Collection Auditor with Docker

The auditor runs in a container; the local LLM (Ollama) runs natively on
your machine. This keeps detection fast on macOS, where a containerised
Ollama would be CPU-only.

## Prerequisites (one time)
1. Install Docker Desktop.
2. Install Ollama (https://ollama.com) and start it.
3. Pull the model:
       ollama pull qwen2.5-coder:7b

## Run the dashboard
    docker compose up --build
Then open http://localhost:8501

## Run the CLI on a repository
Put the repo you want to audit inside the `repos/` folder, then:
    docker compose run --rm auditor python main.py /app/repos/<repo-name>

## Notes
- The model download (~4-5 GB) happens once, via `ollama pull`.
- To audit a repo it must be inside `repos/` so the container can see it.