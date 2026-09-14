# aero-spec-rag: retrieval-grounded aerospace parameter API.
#
# bge-small embeddings (fastembed, CPU) and the ingested corpus are both baked
# in at build time, so the container needs no Ollama, no API key, and no
# network at runtime. AERO_CHROMA_DIR is still writable, so re-running
# `python -m src.ingest` inside a running container works too.

FROM python:3.12-slim AS base

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AERO_EMBEDDINGS=fastembed \
    FASTEMBED_CACHE_PATH=/app/.fastembed \
    AERO_LLM=none

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

# Downloads the embedding model once and bakes the vector store into the image.
RUN python -m src.ingest
ENV HF_HUB_OFFLINE=1

EXPOSE 8001

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=2)" || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8001"]
