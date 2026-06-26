# Container for the RL Data Collection Auditor (the Python app only).
# Ollama runs natively on the host; this image talks to it over HTTP.
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so they cache across rebuilds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project.
COPY . .

# Streamlit UI port.
EXPOSE 8501

# Default: launch the dashboard. Reachable at http://localhost:8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]