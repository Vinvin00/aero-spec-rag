"""Pull structured candidate values out of retrieved chunk text.

The corpus stores its numbers in four-column markdown tables
(`| Quantity | Value | Unit | Notes |`). Chunking can split a table away from
its header, so the parser recognises any four-cell row whose second cell parses
as a number or a numeric range, rather than relying on the header being present.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from .bounds import Quantity, match_doc_alias

_NUMBER = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_RANGE_RE = re.compile(rf"^({_NUMBER})\s*(?:to|-|–|—)\s*({_NUMBER})$")
_SCALAR_RE = re.compile(rf"^({_NUMBER})$")
_SEPARATOR_RE = re.compile(r"^[\s:|-]+$")


@dataclass
class Candidate:
    """One numeric fact recovered from one chunk."""

    quantity_key: str
    label: str
    value: float
    value_low: Optional[float]
    value_high: Optional[float]
    unit: str
    notes: str
    source_doc: str
    source_title: str
    source_type: str
    chunk_index: int
    retrieval_rank: int
    retrieval_score: float

    @property
    def is_range(self) -> bool:
        return self.value_low is not None and self.value_high is not None

    def value_repr(self) -> str:
        if self.is_range:
            return f"{_fmt(self.value_low)} to {_fmt(self.value_high)}"
        return _fmt(self.value)


def _fmt(number: float) -> str:
    text = f"{number:.6g}"
    return text


def _parse_value(cell: str) -> Optional[Tuple[float, Optional[float], Optional[float]]]:
    """Return (representative, low, high) for a scalar or range cell."""
    text = cell.strip().replace(",", "")
    scalar = _SCALAR_RE.match(text)
    if scalar:
        value = float(scalar.group(1))
        return value, None, None
    spread = _RANGE_RE.match(text)
    if spread:
        low, high = float(spread.group(1)), float(spread.group(2))
        if low > high:
            low, high = high, low
        return (low + high) / 2.0, low, high
    return None


def _table_rows(text: str) -> List[List[str]]:
    rows: List[List[str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 4 or any(_SEPARATOR_RE.match(c) for c in cells if c):
            continue
        rows.append(cells)
    return rows


def extract_candidates(
    documents: List[Tuple[Document, float]],
    quantity: Optional[Quantity] = None,
) -> List[Candidate]:
    """Extract candidates from ranked (document, score) pairs.

    If `quantity` is given, only rows matching that quantity are kept.
    """
    candidates: List[Candidate] = []
    for rank, (document, score) in enumerate(documents):
        meta = document.metadata
        for label, raw_value, unit, notes in _table_rows(document.page_content):
            parsed = _parse_value(raw_value)
            if parsed is None:
                continue
            matched = match_doc_alias(label)
            if matched is None:
                continue
            if quantity is not None and matched.key != quantity.key:
                continue
            value, low, high = parsed
            candidates.append(
                Candidate(
                    quantity_key=matched.key,
                    label=label,
                    value=value,
                    value_low=low,
                    value_high=high,
                    unit=unit or matched.canonical_unit,
                    notes=notes,
                    source_doc=meta.get("source_doc", "unknown"),
                    source_title=meta.get("title", ""),
                    source_type=meta.get("source_type", "unknown"),
                    chunk_index=int(meta.get("chunk_index", -1)),
                    retrieval_rank=rank,
                    retrieval_score=score,
                )
            )
    return candidates
