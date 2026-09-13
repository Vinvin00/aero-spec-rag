"""FastAPI wrapper around the grounding graph.

    uvicorn src.api:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .graph import ground_spec
from .schemas import GroundedSpec, GroundSpecRequest

app = FastAPI(
    title="aero-spec-rag",
    version="0.1.0",
    description=(
        "Retrieval-grounded aerospace parameter lookup. All corpus values are "
        "illustrative or textbook-derived; none describe a real system."
    ),
)

# Local development only. Restrict allow_origins before exposing this anywhere.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "collection": config.COLLECTION_NAME,
        "embedding_backend": config.EMBEDDING_BACKEND,
        "llm_backend": config.LLM_BACKEND,
        "llm_model": config.LLM_MODEL if config.LLM_BACKEND != "none" else None,
        "top_k": config.TOP_K,
    }


@app.post("/ground-spec", response_model=GroundedSpec)
def ground_spec_endpoint(request: GroundSpecRequest) -> GroundedSpec:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must not be empty")
    try:
        result = ground_spec(query, top_k=request.top_k)
    except Exception as exc:  # pragma: no cover - surfaced as a 503 to the caller
        raise HTTPException(
            status_code=503,
            detail=(
                f"grounding pipeline unavailable: {exc}. Run `python -m src.ingest`; "
                "if you changed AERO_EMBEDDINGS, the store is dimension-specific and "
                "needs `python -m src.ingest --rebuild`."
            ),
        ) from exc
    return GroundedSpec.model_validate(result)
