# aero-spec-rag: retrieval-grounded aerospace parameter API.
#
# Ships with the deterministic offline backend (no Ollama, no API key, no
# network call at runtime) so the demo works for anyone with just Docker.
# The corpus is ingested once at build time so the container starts serving
# immediately; AERO_CHROMA_DIR is still writable, so re-running `python -m
# src.ingest` inside a running container (e.g. after editing the corpus with
# a bind mount) works too.

FROM python:3.12-slim AS base

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AERO_EMBEDDINGS=local \
    AERO_LLM=none

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

# Bake the vector store into the image: no first-request latency, and no
# dependency on a writable volume just to answer a query.
RUN python -m src.ingest

EXPOSE 8001

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=2)" || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8001"]
