import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Tests use their own store so a developer's .chroma is never mutated.
os.environ.setdefault("AERO_CHROMA_DIR", str(REPO_ROOT / ".chroma-test"))

import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def ingested_store():
    """Build the test vector store once per session."""
    from src.ingest import ingest

    return ingest(rebuild=True)
