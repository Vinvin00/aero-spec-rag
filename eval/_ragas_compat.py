"""Import-time compatibility shim for ragas==0.4.3.

`ragas.llms.base` unconditionally does
`from langchain_community.chat_models.vertexai import ChatVertexAI` (and the
matching `VertexAI` LLM) at import time, for every user, regardless of which
provider they actually evaluate with. That submodule no longer exists in
`langchain-community>=0.3` (moved to the separate `langchain-google-vertexai`
package) -- this repo pins `langchain-community==0.4.2`, so a plain
`import ragas` raises `ModuleNotFoundError` before any of our code runs. This
is a real, reproducible incompatibility between ragas 0.4.3's latest release
and current langchain-community, not a local environment problem: verified by
also trying ragas==0.2.15, which hits the identical import.

Downgrading langchain-community was rejected -- it's a dependency of the core
pipeline (src/ingest.py, src/graph.py), already pinned and tested, and eval is
supposed to sit on top of that pipeline unmodified, not force a different
version of it. Instead, this module registers a harmless dummy module at the
two dead import paths *before* `ragas` is ever imported, which satisfies the
`from ... import` and changes nothing else -- this project uses Ollama or
Anthropic as the ragas judge, never Vertex AI, so the stub is never called.

Import this module first, before importing anything from `ragas`:

    from eval import _ragas_compat  # noqa: F401  (import for side effect)
    import ragas
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

_DEAD_PATHS = {
    "langchain_community.chat_models.vertexai": "ChatVertexAI",
    "langchain_community.llms.vertexai": "VertexAI",
}


def _install() -> None:
    for module_path, symbol in _DEAD_PATHS.items():
        if module_path in sys.modules:
            continue
        stub = types.ModuleType(module_path)
        setattr(stub, symbol, MagicMock(name=f"{symbol}-unavailable-stub"))
        sys.modules[module_path] = stub


_install()
