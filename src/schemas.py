"""Public response schemas for the grounding pipeline."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Citation(BaseModel):
    source_doc: str = Field(description="Corpus filename the value came from.")
    title: str = ""
    source_type: Literal["illustrative", "derived", "unknown"] = "unknown"
    chunk_index: int = -1
    snippet: str = ""
    retrieval_score: float = 0.0


class VerificationFlag(BaseModel):
    code: str
    severity: Literal["info", "warning", "error"]
    message: str


class GroundedSpec(BaseModel):
    """The machine-usable result returned by POST /ground-spec."""

    query: str
    quantity: Optional[str] = Field(
        default=None, description="Registry key for the quantity identified in the query."
    )
    value: Optional[float] = Field(
        default=None, description="Representative value; the midpoint when a range was found."
    )
    value_low: Optional[float] = None
    value_high: Optional[float] = None
    unit: Optional[str] = None
    source_doc: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    verified: bool = False
    answer: str = ""
    context: str = Field(default="", description="Qualifying note attached to the value.")
    plausible_bounds: Optional[List[float]] = None
    flags: List[VerificationFlag] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    narration_model: Optional[str] = Field(
        default=None,
        description=(
            "Set only when an LLM rephrased `answer`. Structured fields are always "
            "produced deterministically."
        ),
    )
    disclaimer: str = (
        "Illustrative corpus. Values are teaching or order-of-magnitude figures, "
        "not the specifications of any real system."
    )


class GroundSpecRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=25)
