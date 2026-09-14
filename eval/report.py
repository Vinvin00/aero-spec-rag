"""Run the full eval and write a timestamped Markdown report + latest.json.

    python -m eval.report

Writes:
  eval/results/report_<timestamp>.md   human-readable, one per run
  eval/results/latest.json             machine-readable, overwritten each run
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .run_eval import run_full_eval

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# See DECISIONS.md for why these specific numbers. Applied to the 12 "good"
# questions only -- trap questions have no valid answer/context by
# construction, so RAGAS scores on them are informational, not gated.
THRESHOLDS = {
    "faithfulness": 0.80,
    "context_precision": 0.70,
    "context_recall": 0.70,
    "answer_relevancy": 0.70,
}
VERIFY_NODE_ACCURACY_THRESHOLD = 1.0


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _metric_table(aggregate: Dict[str, float]) -> str:
    lines = ["| Metric | Score | Threshold | Result |", "| --- | --- | --- | --- |"]
    for metric, threshold in THRESHOLDS.items():
        score = aggregate.get(metric, 0.0)
        ok = "PASS" if score >= threshold else "FAIL"
        lines.append(f"| {metric} | {_fmt(score)} | ≥ {threshold} | {ok} |")
    return "\n".join(lines)


def _per_question_table(per_question) -> str:
    lines = [
        "| # | Trap | Question | Verified (exp/actual) | Faith. | Ctx.Prec. | Ctx.Rec. | Ans.Rel. | Verdict |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, q in enumerate(per_question, 1):
        trap = "yes" if q["is_trap"] else ""
        verified = f"{q['expected_verified']}/{q['verified']}"
        if q["is_trap"]:
            verdict = "OK" if q["verified"] == q["expected_verified"] else "WRONG"
        else:
            metric_pass = all(
                q[metric] >= threshold for metric, threshold in THRESHOLDS.items()
            )
            verify_ok = q["verified"] == q["expected_verified"]
            verdict = "PASS" if (metric_pass and verify_ok) else "FAIL"
        question = q["question"].replace("|", "\\|")
        lines.append(
            f"| {i} | {trap} | {question} | {verified} | "
            f"{_fmt(q['faithfulness'])} | {_fmt(q['context_precision'])} | "
            f"{_fmt(q['context_recall'])} | {_fmt(q['answer_relevancy'])} | {verdict} |"
        )
    return "\n".join(lines)


def _overall_verdict(eval_result: Dict[str, Any]) -> str:
    aggregate = eval_result["aggregate_metrics"]
    metrics_pass = all(aggregate.get(m, 0.0) >= t for m, t in THRESHOLDS.items())
    verify_pass = eval_result["verify_node_accuracy"]["accuracy"] >= VERIFY_NODE_ACCURACY_THRESHOLD
    if metrics_pass and verify_pass:
        return "PASS -- all RAGAS thresholds met on good questions, verify_node_accuracy is 1.0."
    failed = [m for m, t in THRESHOLDS.items() if aggregate.get(m, 0.0) < t]
    parts = []
    if failed:
        parts.append(f"RAGAS metric(s) below threshold: {', '.join(failed)}")
    if not verify_pass:
        acc = eval_result["verify_node_accuracy"]["accuracy"]
        parts.append(f"verify_node_accuracy {acc:.3f} < {VERIFY_NODE_ACCURACY_THRESHOLD}")
    return "FAIL -- " + "; ".join(parts)


def render_markdown(eval_result: Dict[str, Any], timestamp: str) -> str:
    verify_acc = eval_result["verify_node_accuracy"]
    lines = [
        f"# aero-spec-rag eval report -- {timestamp}",
        "",
        f"Judge LLM: `{eval_result['judge_llm']}`. "
        f"{eval_result['good_questions']} good questions, "
        f"{eval_result['trap_questions']} trap questions "
        f"({eval_result['testset_size']} total).",
        "",
        "## RAGAS metrics (good questions only, n="
        f"{eval_result['good_questions']})",
        "",
        _metric_table(eval_result["aggregate_metrics"]),
        "",
        "## verify_node_accuracy (custom, not a RAGAS metric)",
        "",
        f"{verify_acc['correct']} / {verify_acc['total']} cases matched their "
        f"expected `verified` flag ({_fmt(verify_acc['accuracy'])}). Threshold: "
        f"{VERIFY_NODE_ACCURACY_THRESHOLD}. Covers both the 12 good questions "
        "(expect `verified: true`) and the 3 trap questions (expect "
        "`verified: false`) -- this is the signal for whether the verify node "
        "correctly refuses to fabricate a value for out-of-corpus questions.",
        "",
        "## Per-question breakdown",
        "",
        _per_question_table(eval_result["per_question"]),
        "",
        "Trap rows are not held to the RAGAS thresholds (see DECISIONS.md) -- "
        "their verdict is OK/WRONG against `verify_node_accuracy` alone, since "
        "a trap question has no valid answer or context by construction.",
        "",
        "## Verdict",
        "",
        _overall_verdict(eval_result),
        "",
    ]
    return "\n".join(lines)


def run_and_report() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    eval_result = run_full_eval()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    report_path = RESULTS_DIR / f"report_{timestamp}.md"
    report_path.write_text(render_markdown(eval_result, timestamp))

    latest_path = RESULTS_DIR / "latest.json"
    latest_payload = {
        "timestamp": timestamp,
        "thresholds": THRESHOLDS,
        "verify_node_accuracy_threshold": VERIFY_NODE_ACCURACY_THRESHOLD,
        "overall_verdict": _overall_verdict(eval_result),
        **eval_result,
    }
    latest_path.write_text(json.dumps(latest_payload, indent=2, default=str))

    return report_path


if __name__ == "__main__":
    path = run_and_report()
    print(f"wrote {path}")
    print(f"wrote {RESULTS_DIR / 'latest.json'}")
