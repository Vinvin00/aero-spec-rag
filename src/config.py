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

# "local" (default, offline deterministic), "openai", or "voyage".
EMBEDDING_BACKEND = os.environ.get("AERO_EMBEDDINGS", "local").lower()
EMBEDDING_DIM = int(os.environ.get("AERO_EMBEDDING_DIM", 512))
