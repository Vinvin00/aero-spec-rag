"""Optional LLM backend.

The pipeline is deterministic by default (`AERO_LLM=none`): every numeric field
in a response is parsed out of a corpus table and bounds-checked, with no model
in the loop. Enabling an LLM adds exactly two capabilities, neither of which can
invent a number:

* `classify_quantity` — a fallback for queries the alias registry misses. It may
  only return a key that already exists in the registry; the value still comes
  from a parsed, verified table row.
* `narrate` — rewrites the `answer` sentence in natural language from fields
  that have already been selected and verified. The structured fields are never
  touched, and a failure falls back to the deterministic sentence.

Set AERO_LLM=ollama to run locally with no API key (AERO_LLM_MODEL defaults to
llama3.1:8b), or AERO_LLM=anthropic to use a hosted model.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from . import config


def llm_enabled() -> bool:
    return config.LLM_BACKEND != "none"


@lru_cache(maxsize=1)
def get_llm():
    """Return a chat model, or None when no LLM backend is configured."""
    backend = config.LLM_BACKEND
    if backend == "none":
        return None
    if backend == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.LLM_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.0,
            client_kwargs={"timeout": config.LLM_TIMEOUT},
        )
    if backend == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=config.LLM_MODEL, temperature=0.0)
    raise ValueError(f"Unknown AERO_LLM backend: {config.LLM_BACKEND!r}")


_CLASSIFY_PROMPT = """You label aerospace parameter questions.

Reply with exactly one key from this list, and nothing else. If none of them is \
what the question asks for, reply NONE.

{keys}

Question: {query}
Key:"""

_NARRATE_PROMPT = """Restate this looked-up engineering value as one plain sentence.

Rules:
- Use the value, unit and context exactly as given. Do not round, convert, \
recompute, or introduce any number that is not listed below.
- One sentence. No preamble, no markdown, no citation.

quantity: {quantity}
value: {value}
unit: {unit}
context: {context}

Sentence:"""


def classify_quantity(query: str, keys: List[str]) -> Optional[str]:
    """Map a query onto a registry key. Returns None if the model declines."""
    model = get_llm()
    if model is None:
        return None
    try:
        response = model.invoke(
            _CLASSIFY_PROMPT.format(keys="\n".join(keys), query=query)
        )
    except Exception:
        return None
    answer = str(getattr(response, "content", "")).strip().strip(".`\"' ").lower()
    # Constrain hard: the model may only pick something already in the registry.
    return answer if answer in set(keys) else None


def narrate(
    quantity: str,
    value: str,
    unit: str,
    context: str,
    source_doc: str,
) -> Optional[str]:
    """Phrase a verified value as a sentence. Returns None on any failure."""
    model = get_llm()
    if model is None:
        return None
    try:
        response = model.invoke(
            _NARRATE_PROMPT.format(
                quantity=quantity.replace("_", " "),
                value=value,
                unit=unit,
                context=context or "none given",
            )
        )
    except Exception:
        return None
    sentence = " ".join(str(getattr(response, "content", "")).split()).strip(" `\"'")
    if not sentence:
        return None
    # Hard guard: the verified value must survive into the prose verbatim,
    # otherwise the deterministic sentence is kept instead.
    if value not in sentence:
        return None
    if not sentence.endswith("."):
        sentence += "."
    # The citation is appended deterministically rather than asked of the model,
    # so an answer can never lose its provenance to a phrasing slip.
    return f"{sentence} Source: {source_doc}."
