"""Central configuration for the aero-spec-rag pipeline."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent

CORPUS_DIR = Path(os.environ.get("AERO_CORPUS_DIR", PACKAGE_ROOT / "corpus"))
CHROMA_DIR = Path(os.environ.get("AERO_CHROMA_DIR", REPO_ROOT / ".chroma"))
COLLECTION_NAME = os.environ.get("AERO_COLLECTION", "aero_spec")

CHUNK_SIZE = int(os.environ.get("AERO_CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.environ.get("AERO_CHUNK_OVERLAP", 100))
TOP_K = int(os.environ.get("AERO_TOP_K", 5))

# "fastembed" (default, learned, local CPU), "local" (hashing, zero download), or "ollama".
EMBEDDING_BACKEND = os.environ.get("AERO_EMBEDDINGS", "fastembed").lower()
FASTEMBED_MODEL = os.environ.get("AERO_FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = int(os.environ.get("AERO_EMBEDDING_DIM", 512))

# --- Ollama / LLM ---------------------------------------------------------
# "none" (default, fully deterministic), "ollama", or "anthropic".
LLM_BACKEND = os.environ.get("AERO_LLM", "none").lower()
DEFAULT_LLM_MODELS = {"ollama": "llama3.1:8b", "anthropic": "claude-haiku-4-5-20251001"}
LLM_MODEL = os.environ.get("AERO_LLM_MODEL")  # None -> per-backend default above
LLM_TIMEOUT = float(os.environ.get("AERO_LLM_TIMEOUT", 30))

OLLAMA_BASE_URL = os.environ.get("AERO_OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.environ.get("AERO_OLLAMA_EMBED_MODEL", "nomic-embed-text")
