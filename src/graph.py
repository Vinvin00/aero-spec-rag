"""The grounding graph (LangGraph StateGraph).

    retrieve --(quantity unresolved)--------------------------> decline
    retrieve --(quantity resolved)--> verify
    verify   --(a candidate survived the bounds check)--------> propose
    verify   --(none survived, first attempt)--> widen --> retrieve
    verify   --(none survived, already widened)---------------> decline
    propose, decline --> finalize --> END

See DECISIONS.md.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from . import config
from .bounds import BY_KEY, QUANTITIES, Quantity, match_query
from .embeddings import tokenize
from .extract import Candidate, extract_candidates
from .ingest import get_vectorstore
from .llm import classify_quantity, llm_enabled, model_name, narrate
from .schemas import Citation, GroundedSpec, VerificationFlag

# Query terms that carry no retrieval signal.
_STOPWORDS = {
    "a", "an", "the", "what", "whats", "is", "are", "of", "for", "in", "at", "to",
    "and", "or", "on", "me", "my", "give", "tell", "about", "value", "values",
    "typical", "should", "i", "use", "does", "do", "how", "much", "many", "be",
}

_UNIT_SCALES = {
    "km": 1000.0, "kilometre": 1000.0, "kilometer": 1000.0,
    "m": 1.0, "metre": 1.0, "meter": 1.0,
    "ft": 0.3048, "feet": 0.3048,
}
_ALTITUDE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(km|kilometres?|kilometers?|m|metres?|meters?|ft|feet)\b",
    re.IGNORECASE,
)


class GroundingState(TypedDict, total=False):
    """State carried between nodes."""

    query: str
    top_k: int
    attempts: int
    quantity_key: Optional[str]
    retrieved: List[Dict[str, Any]]
    candidates: List[Candidate]
    flags: List[VerificationFlag]
    selected: Optional[Candidate]
    result: Dict[str, Any]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def query_terms(query: str) -> List[str]:
    """Content tokens, expanded with unit-normalised altitude forms.

    "density at 10 km" also yields "10000", so it can match a table note
    reading "ISA at 10000 m altitude".
    """
    terms = [t for t in tokenize(query) if t not in _STOPWORDS]
    for raw, unit in _ALTITUDE_RE.findall(query):
        unit = unit.lower()
        metres = float(raw) * _UNIT_SCALES.get(unit, _UNIT_SCALES.get(unit.rstrip("s")))
        terms.append(f"{metres:.6g}")
        if metres.is_integer():
            terms.append(str(int(metres)))
    return terms


def lexical_score(text: str, terms: List[str]) -> float:
    """Fraction of query terms present in the text."""
    if not terms:
        return 0.0
    haystack = set(tokenize(text))
    return sum(1.0 for t in terms if t in haystack) / len(terms)


@lru_cache(maxsize=1)
def _default_store():
    return get_vectorstore()


# --------------------------------------------------------------------------
# nodes
# --------------------------------------------------------------------------


def retrieve_node(state: GroundingState, store=None) -> GroundingState:
    """Vector search, lexically rescored, truncated to top_k."""
    query = state["query"]
    top_k = int(state.get("top_k") or config.TOP_K)
    vectorstore = store or _default_store()

    # Identify the quantity first, so it can steer retrieval. The alias registry
    # is exact-match; when it misses, an LLM (if configured) can recognise the
    # phrasing, but may only return a key that already exists in the registry.
    matched = match_query(query)
    quantity_key = matched.key if matched else None
    if quantity_key is None and llm_enabled():
        quantity_key = classify_quantity(query, [q.key for q in QUANTITIES])

    # Expand the search with the quantity's canonical vocabulary. "How heavy is
    # the bird at launch?" shares no words with a "launch mass" table row, so
    # without this the right chunk never enters the pool.
    quantity = BY_KEY.get(quantity_key or "")
    alias_text = " ".join(quantity.doc_aliases) if quantity else ""
    search_text = f"{query} {alias_text}".strip()

    # Hybrid retrieval: over-fetch by vector similarity, then rescore lexically.
    # Exact numeric tokens ("10000", "0.47") matter here and dense embedders blur them.
    pool: List[Tuple[Document, float]] = vectorstore.similarity_search_with_score(
        search_text, k=max(top_k * 4, top_k)
    )
    terms = query_terms(query)
    alias_terms = query_terms(alias_text)
    ranked = sorted(
        pool,
        key=lambda pair: (
            lexical_score(pair[0].page_content, terms)
            + 0.3 * lexical_score(pair[0].page_content, alias_terms)
            - 0.15 * float(pair[1])
        ),
        reverse=True,
    )[:top_k]

    return {
        "quantity_key": quantity_key,
        "retrieved": [
            {
                "content": doc.page_content,
                "metadata": dict(doc.metadata),
                # Chroma returns a distance; report similarity for readability.
                "score": round(1.0 / (1.0 + float(distance)), 4),
                "distance": round(float(distance), 4),
            }
            for doc, distance in ranked
        ],
    }


def verify_node(state: GroundingState) -> GroundingState:
    """Extract candidate values and check them against the bounds table."""
    quantity: Quantity = BY_KEY[state["quantity_key"]]
    pairs = [
        (Document(page_content=r["content"], metadata=r["metadata"]), r["score"])
        for r in state.get("retrieved", [])
    ]
    flags: List[VerificationFlag] = []
    if state.get("attempts", 0) > 0:
        flags.append(
            VerificationFlag(
                code="widened_retrieval",
                severity="info",
                message=(
                    "No candidate survived verification at the requested top_k, so "
                    f"retrieval was re-run with top_k={state.get('top_k')}."
                ),
            )
        )

    candidates = extract_candidates(pairs, quantity)

    if not candidates:
        flags.append(
            VerificationFlag(
                code="no_candidate_values",
                severity="error",
                message=(
                    f"No tabulated value for {quantity.key!r} was found in the retrieved "
                    "chunks."
                ),
            )
        )

    kept: List[Candidate] = []
    for candidate in candidates:
        spec = BY_KEY[candidate.quantity_key]
        checks = [candidate.value]
        if candidate.is_range:
            checks = [candidate.value_low, candidate.value_high]
        if all(spec.contains(v) for v in checks):
            kept.append(candidate)
        else:
            flags.append(
                VerificationFlag(
                    code="out_of_bounds",
                    severity="error",
                    message=(
                        f"Discarded {candidate.value_repr()} {candidate.unit} for "
                        f"{spec.key} from {candidate.source_doc}: outside the plausible "
                        f"range {spec.low} to {spec.high} {spec.canonical_unit}."
                    ),
                )
            )

    for candidate in kept:
        spec = BY_KEY[candidate.quantity_key]
        if candidate.unit and spec.canonical_unit not in candidate.unit:
            flags.append(
                VerificationFlag(
                    code="unit_mismatch",
                    severity="warning",
                    message=(
                        f"{candidate.source_doc} reports {spec.key} in {candidate.unit!r}; "
                        f"the registry expects {spec.canonical_unit!r}."
                    ),
                )
            )

    scalars = [c.value for c in kept]
    if len(scalars) > 1 and min(scalars) > 0 and max(scalars) / min(scalars) > 10:
        flags.append(
            VerificationFlag(
                code="wide_spread",
                severity="warning",
                message=(
                    "Retrieved values for this quantity span more than an order of "
                    "magnitude; the query may be under-specified (for example, missing "
                    "an altitude or a body shape)."
                ),
            )
        )

    return {"candidates": kept, "flags": flags}


def _selection_score(candidate: Candidate, terms: List[str]) -> float:
    """Rank candidates by how well their row context answers the query."""
    context = f"{candidate.label} {candidate.notes} {candidate.source_title}"
    lexical = lexical_score(context, terms)
    rank_bonus = 1.0 / (1.0 + candidate.retrieval_rank)
    # Prefer a derived (physics-backed) row over an illustrative placeholder.
    provenance = 0.1 if candidate.source_type == "derived" else 0.0
    return 2.0 * lexical + 0.5 * rank_bonus + provenance


def _citations(state: GroundingState) -> List[Citation]:
    return [
        Citation(
            source_doc=r["metadata"].get("source_doc", "unknown"),
            title=r["metadata"].get("title", ""),
            source_type=r["metadata"].get("source_type", "unknown"),
            chunk_index=int(r["metadata"].get("chunk_index", -1)),
            snippet=r["content"][:300].strip(),
            retrieval_score=r["score"],
        )
        for r in state.get("retrieved", [])
    ]


def widen_node(state: GroundingState) -> GroundingState:
    """Double top_k before a second retrieval pass."""
    top_k = int(state.get("top_k") or config.TOP_K)
    return {"top_k": top_k * 2, "attempts": state.get("attempts", 0) + 1}


def decline_node(state: GroundingState) -> GroundingState:
    """Return citations only, never a value, when nothing could be verified."""
    quantity = BY_KEY.get(state.get("quantity_key") or "")
    flags: List[VerificationFlag] = list(state.get("flags", []))
    if quantity is None and not any(f.code == "unknown_quantity" for f in flags):
        flags.append(
            VerificationFlag(
                code="unknown_quantity",
                severity="warning",
                message=(
                    "The query did not match any quantity in the bounds registry, so no "
                    "numeric verification was possible."
                ),
            )
        )
    spec = GroundedSpec(
        query=state["query"],
        quantity=quantity.key if quantity else None,
        verified=False,
        confidence=0.0,
        answer=(
            "No grounded value could be extracted from the corpus for this query. "
            "The retrieved passages are cited so the question can be narrowed."
        ),
        plausible_bounds=[quantity.low, quantity.high] if quantity else None,
        flags=flags,
        citations=_citations(state),
    )
    return {"selected": None, "flags": flags, "result": spec.model_dump()}


def propose_node(state: GroundingState) -> GroundingState:
    """Pick the best surviving candidate and shape the structured response."""
    query = state["query"]
    terms = query_terms(query)
    candidates: List[Candidate] = state["candidates"]
    flags: List[VerificationFlag] = list(state.get("flags", []))
    citations = _citations(state)

    best = max(candidates, key=lambda c: _selection_score(c, terms))
    match_strength = lexical_score(f"{best.label} {best.notes}", terms)

    confidence = 0.35 + 0.45 * match_strength + 0.2 / (1.0 + best.retrieval_rank)
    if best.source_type == "illustrative":
        confidence -= 0.10
    if any(f.severity == "warning" for f in flags):
        confidence -= 0.10
    if best.is_range:
        confidence -= 0.05
    confidence = round(max(0.0, min(1.0, confidence)), 3)

    if best.source_type == "illustrative":
        flags.append(
            VerificationFlag(
                code="illustrative_source",
                severity="info",
                message=(
                    f"{best.source_doc} is labelled illustrative: the value is an "
                    "order-of-magnitude placeholder, not a measured or derived figure."
                ),
            )
        )

    spec_bounds = BY_KEY[best.quantity_key]
    answer = (
        f"{spec_bounds.key.replace('_', ' ')} = {best.value_repr()} {best.unit}"
        f"{' (' + best.notes + ')' if best.notes else ''}, from {best.source_doc}."
    )

    result = GroundedSpec(
        query=query,
        quantity=best.quantity_key,
        value=round(best.value, 10),
        value_low=best.value_low,
        value_high=best.value_high,
        unit=best.unit,
        source_doc=best.source_doc,
        confidence=confidence,
        verified=True,
        answer=answer,
        context=best.notes,
        plausible_bounds=[spec_bounds.low, spec_bounds.high],
        flags=flags,
        citations=citations,
    )
    return {"selected": best, "flags": flags, "result": result.model_dump()}


def finalize_node(state: GroundingState) -> GroundingState:
    """Validate the payload, and optionally phrase the answer with an LLM.

    Narration is presentation only: it rewrites `answer` from fields that are
    already selected and verified, and is rejected unless the value and the
    source document both survive into the sentence. Every structured field is
    produced deterministically regardless of backend.
    """
    spec = GroundedSpec.model_validate(state.get("result") or {})

    selected: Optional[Candidate] = state.get("selected")
    if selected is not None and llm_enabled():
        sentence = narrate(
            quantity=spec.quantity or selected.quantity_key,
            value=selected.value_repr(),
            unit=spec.unit or "",
            context=spec.context,
            source_doc=spec.source_doc or "",
        )
        if sentence:
            spec.answer = sentence
            spec.narration_model = f"{config.LLM_BACKEND}:{model_name()}"
        else:
            spec.flags.append(
                VerificationFlag(
                    code="narration_unavailable",
                    severity="info",
                    message=(
                        "LLM narration was skipped or rejected; the deterministic "
                        "answer sentence is reported instead."
                    ),
                )
            )

    return {"result": spec.model_dump()}


# --------------------------------------------------------------------------
# graph
# --------------------------------------------------------------------------


MAX_RETRIEVAL_ATTEMPTS = 2


def route_after_retrieve(state: GroundingState) -> str:
    return "verify" if state.get("quantity_key") else "decline"


def route_after_verify(state: GroundingState) -> str:
    if state.get("candidates"):
        return "propose"
    if state.get("attempts", 0) + 1 < MAX_RETRIEVAL_ATTEMPTS:
        return "widen"
    return "decline"


def compile_graph(store=None):
    builder = StateGraph(GroundingState)
    builder.add_node("retrieve", lambda s: retrieve_node(s, store=store))
    builder.add_node("verify", verify_node)
    builder.add_node("widen", widen_node)
    builder.add_node("propose", propose_node)
    builder.add_node("decline", decline_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "retrieve")
    builder.add_conditional_edges("retrieve", route_after_retrieve, ["verify", "decline"])
    builder.add_conditional_edges("verify", route_after_verify, ["propose", "widen", "decline"])
    builder.add_edge("widen", "retrieve")
    builder.add_edge("propose", "finalize")
    builder.add_edge("decline", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()


class _Pipeline:
    """Thin handle over the compiled graph; also the eval harness's spy point."""

    def __init__(self, store=None):
        self.graph = compile_graph(store)

    def invoke(self, state: GroundingState) -> GroundingState:
        return self.graph.invoke(state)


def build_graph(store=None):
    """Build the grounding graph. Pass `store` to inject a test vector store."""
    return _Pipeline(store)


@lru_cache(maxsize=1)
def get_graph():
    return build_graph()


def ground_spec(query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
    """Run one query through the graph and return the structured result."""
    state: GroundingState = {"query": query}
    if top_k:
        state["top_k"] = top_k
    return get_graph().invoke(state)["result"]


if __name__ == "__main__":
    import json
    import sys

    question = " ".join(sys.argv[1:]) or "What drag coefficient should I use for a sphere?"
    print(json.dumps(ground_spec(question), indent=2))
