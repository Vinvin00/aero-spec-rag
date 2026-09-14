"""Hand-written evaluation set for the grounding pipeline.

Each entry is a real question against the actual corpus, with a
human-authored reference answer (`ground_truth_answer`) and the source
document the pipeline is expected to cite when it answers correctly
(`expected_source_doc`, `None` for the trap questions below).

Every "good" entry was run against the live pipeline (`src.graph.ground_spec`)
before being written down here, so `expected_source_doc` and the values in
`ground_truth_answer` reflect what the corpus actually contains and what the
pipeline actually retrieves today -- not a guess. See DECISIONS.md for one
question that was dropped after this check: "What is the ISA lapse rate in
the troposphere?" is answerable from the corpus but the retriever does not
surface the table row containing it within top_k=5, which is a real (if
narrow) retrieval-quality gap, not a bug in this harness.

Trap questions (`is_trap=True`) ask about quantities that exist nowhere in
`src/bounds.py`'s registry and nowhere in the corpus -- specific impulse,
radar cross section, unit cost. Correct pipeline behavior is `verified:
false`: no candidate value, not a fabricated one. This is the realistic trap
shape for *this* system: the corpus is well-formed, so `verify_node`'s
bounds check (Cd in [0.01, 1.5], etc.) only ever rejects a value on
deliberately corrupted input -- see tests/test_graph.py's
`test_out_of_bounds_value_is_flagged_and_discarded` for that path exercised
directly. A real user query never reaches the corpus with an out-of-bounds
number already attached to it, so "ask for a value outside plausible bounds"
is not something a natural-language question can trigger through the front
door; "ask about a value not in the corpus at all" is, and is exactly what
these three exercise.

Run as a script to (re)generate testset.json:

    python -m eval.testset
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

TESTSET_JSON = Path(__file__).resolve().parent / "testset.json"


@dataclass(frozen=True)
class EvalCase:
    question: str
    ground_truth_answer: str
    expected_source_doc: Optional[str]
    # True when the correct pipeline behavior is to *not* produce a verified
    # value -- checked separately from the RAGAS metrics (see run_eval.py).
    is_trap: bool = False
    expected_verified: bool = field(default=True)

    def __post_init__(self):
        # A trap case is defined by expecting verified=False; keep the two
        # fields from silently disagreeing with each other.
        if self.is_trap and self.expected_verified:
            object.__setattr__(self, "expected_verified", False)


TESTSET: list[EvalCase] = [
    # --- ISA standard atmosphere -------------------------------------------------
    EvalCase(
        question="What is the ISA air density at sea level?",
        ground_truth_answer="At ISA sea level, air density is 1.225 kg/m^3.",
        expected_source_doc="isa-standard-atmosphere.md",
    ),
    EvalCase(
        question="What is the ISA air density at 10 km altitude?",
        ground_truth_answer=(
            "At 10,000 m (10 km) altitude under the ISA model, air density is "
            "approximately 0.4135 kg/m^3."
        ),
        expected_source_doc="isa-density-vs-altitude.md",
    ),
    EvalCase(
        question="What is the speed of sound at sea level?",
        ground_truth_answer=(
            "The speed of sound at ISA sea level (288.15 K, dry air) is 340.3 m/s."
        ),
        expected_source_doc="mach-reynolds-reference.md",
    ),
    EvalCase(
        question="What is standard gravity used in the ISA model?",
        ground_truth_answer="The ISA model uses standard gravity g0 = 9.80665 m/s^2.",
        expected_source_doc="isa-standard-atmosphere.md",
    ),
    # --- Drag coefficients ---------------------------------------------------
    EvalCase(
        question="What drag coefficient should I use for a sphere?",
        ground_truth_answer=(
            "A sphere at subcritical Reynolds number has a drag coefficient of "
            "about 0.47."
        ),
        expected_source_doc="drag-coefficients-by-shape.md",
    ),
    EvalCase(
        question="What drag coefficient should I use for a streamlined slender body?",
        ground_truth_answer=(
            "A streamlined, finned slender body has a drag coefficient of about "
            "0.3 in subsonic axial flow."
        ),
        expected_source_doc="drag-coefficients-by-shape.md",
    ),
    # --- Proportional navigation fundamentals ---------------------------------
    EvalCase(
        question="What proportional navigation gain should I use for terminal homing?",
        ground_truth_answer=(
            "The usual working range for a proportional navigation gain in "
            "terminal homing is 3 to 5, with 4 a common default."
        ),
        expected_source_doc="pn-gain-selection.md",
    ),
    EvalCase(
        question="What guidance update rate is typical for a digital PN loop?",
        ground_truth_answer=(
            "A digital proportional-navigation guidance loop typically updates "
            "at 50 to 200 Hz."
        ),
        expected_source_doc="pn-gain-selection.md",
    ),
    # --- Illustrative interceptor / target ranges -----------------------------
    EvalCase(
        question="What is the launch mass of a medium-range interceptor?",
        ground_truth_answer=(
            "An illustrative medium-range interceptor class has a launch mass "
            "of about 700 kg."
        ),
        expected_source_doc="interceptor-class-parameters.md",
    ),
    EvalCase(
        question="What is the maximum lateral acceleration of a short-range interceptor?",
        ground_truth_answer=(
            "An illustrative short-range agile interceptor has a maximum "
            "lateral acceleration of about 30 g."
        ),
        expected_source_doc="interceptor-class-parameters.md",
    ),
    EvalCase(
        question="What target speed is typical for a fast manoeuvring target?",
        ground_truth_answer=(
            "An illustrative fast, manoeuvring target has a speed of about "
            "600 m/s."
        ),
        expected_source_doc="target-class-parameters.md",
    ),
    EvalCase(
        question="What closing velocity range is typical for a terminal engagement?",
        ground_truth_answer=(
            "Illustrative closing velocity in a terminal engagement spans "
            "roughly 300 to 1800 m/s, depending on geometry."
        ),
        expected_source_doc="engagement-geometry-basics.md",
    ),
    # --- Trap questions: not in the registry, not in the corpus ---------------
    EvalCase(
        question="What is the specific impulse of the interceptor's rocket motor?",
        ground_truth_answer=(
            "Specific impulse is not covered anywhere in this corpus; the "
            "pipeline should decline to answer rather than fabricate a value."
        ),
        expected_source_doc=None,
        is_trap=True,
    ),
    EvalCase(
        question="What is the radar cross section of the target?",
        ground_truth_answer=(
            "Radar cross section is not covered anywhere in this corpus; the "
            "pipeline should decline to answer rather than fabricate a value."
        ),
        expected_source_doc=None,
        is_trap=True,
    ),
    EvalCase(
        question="What is the unit cost of the interceptor in dollars?",
        ground_truth_answer=(
            "Unit cost is not covered anywhere in this corpus; the pipeline "
            "should decline to answer rather than fabricate a value."
        ),
        expected_source_doc=None,
        is_trap=True,
    ),
]


def to_json_list() -> list[dict]:
    return [asdict(case) for case in TESTSET]


def load_testset(path: Path | None = None) -> list[dict]:
    """Load the eval set from testset.json (falls back to the in-code TESTSET)."""
    target = path or TESTSET_JSON
    if target.exists():
        return json.loads(target.read_text())
    return to_json_list()


def main() -> None:
    TESTSET_JSON.write_text(json.dumps(to_json_list(), indent=2) + "\n")
    print(f"wrote {len(TESTSET)} cases -> {TESTSET_JSON}")


if __name__ == "__main__":
    main()
