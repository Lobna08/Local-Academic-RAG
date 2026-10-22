# One image serves both the API and the UI; the command is set per service in
# docker-compose.yml. Ollama stays on the host and the Chroma index is mounted,
# so data/ is not copied: ingestion never runs inside the container.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY ui/ ui/
