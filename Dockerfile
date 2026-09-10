# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# System dependencies required by chromadb / pypdf
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Ollama runs as a SEPARATE service (the LLM is local, not bundled here).
# Point OLLAMA_API_BASE at the host Ollama instance, e.g.:
#   docker run -e OLLAMA_API_BASE=http://host.docker.internal:11434 -p 8501:8501 app
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]