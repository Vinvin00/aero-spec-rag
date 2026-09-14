"""Eval harness tests: fast, free, no judge LLM involved.

`collect_results` runs the *real* pipeline (retrieve/verify/propose/finalize)
end to end -- that part is never mocked, per the task brief. What's mocked
here is only the expensive half: the RAGAS judge LLM call in `score_with_ragas`
/ `run_full_eval`, which needs a live Ollama (or Anthropic) model and takes
minutes -- unsuitable for a test that should run in CI on every commit. Real
metric computation against a live judge is `eval/report.py`'s job, run
manually.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from eval.run_eval import collect_results, run_full_eval, verify_node_accuracy
from eval.testset import TESTSET_JSON, load_testset

REQUIRED_FIELDS = {
    "question",
    "ground_truth_answer",
    "expected_source_doc",
    "is_trap",
    "expected_verified",
}


# --------------------------------------------------------------------------
# testset.json schema
# --------------------------------------------------------------------------


def test_testset_json_exists_and_is_inspectable():
    assert TESTSET_JSON.exists(), "testset.json must be committed, not generated on the fly"
    data = json.loads(TESTSET_JSON.read_text())
    assert isinstance(data, list)


def test_testset_size_is_in_the_required_range():
    data = load_testset()
    assert 10 <= len(data) <= 15


def test_testset_schema():
    data = load_testset()
    for case in data:
        assert REQUIRED_FIELDS <= case.keys()
        assert isinstance(case["question"], str) and case["question"].strip()
        assert isinstance(case["ground_truth_answer"], str) and case["ground_truth_answer"].strip()
        assert case["expected_source_doc"] is None or isinstance(case["expected_source_doc"], str)
        assert isinstance(case["is_trap"], bool)
        assert isinstance(case["expected_verified"], bool)


def test_testset_has_at_least_three_traps():
    data = load_testset()
    traps = [c for c in data if c["is_trap"]]
    assert len(traps) >= 3
    # A trap's whole point is that verification should fail.
    assert all(c["expected_verified"] is False for c in traps)


def test_testset_covers_the_required_topics():
    """Loose coverage check: every named topic area has at least one question."""
    data = load_testset()
    docs = {c["expected_source_doc"] for c in data if c["expected_source_doc"]}
    assert any("isa" in d for d in docs), "no ISA atmosphere question"
    assert any("drag" in d for d in docs), "no drag coefficient question"
    assert any("pn-gain" in d or "proportional-navigation" in d for d in docs), "no PN question"
    assert any("interceptor" in d or "target" in d for d in docs), "no interceptor/target question"


# --------------------------------------------------------------------------
# collect_results: the real pipeline, no LLM judge
# --------------------------------------------------------------------------


def test_collect_results_runs_the_real_pipeline_end_to_end():
    data = load_testset()[:2]
    results = collect_results(data)

    assert len(results) == 2
    for case, result in zip(data, results):
        assert result.question == case["question"]
        # The pipeline actually ran: it produced *some* retrieval, even for
        # a question that ends up unverified.
        assert isinstance(result.retrieved_contexts, list)
        assert result.generated_answer  # never empty -- always at least the decline message


def test_collect_results_never_passes_ground_truth_into_the_pipeline(monkeypatch):
    """The one hard requirement from the task brief: no leakage."""
    seen_queries = []
    from src import graph as graph_module

    original_invoke = graph_module._Pipeline.invoke

    def spy_invoke(self, state):
        seen_queries.append(state.get("query"))
        return original_invoke(self, state)

    monkeypatch.setattr(graph_module._Pipeline, "invoke", spy_invoke)

    data = load_testset()[:2]
    collect_results(data)

    for query, case in zip(seen_queries, data):
        assert query == case["question"]
        assert case["ground_truth_answer"] not in query


def test_good_questions_verify_and_trap_questions_do_not():
    """A quick sanity pass on the real pipeline's behavior (not a metrics test)."""
    data = load_testset()
    good = next(c for c in data if not c["is_trap"])
    trap = next(c for c in data if c["is_trap"])

    results = collect_results([good, trap])
    assert results[0].verified is True
    assert results[1].verified is False


def test_verify_node_accuracy_scores_correct_and_incorrect_cases():
    data = load_testset()[:2]
    results = collect_results(data)
    acc = verify_node_accuracy(results)

    assert acc["total"] == 2
    assert 0.0 <= acc["accuracy"] <= 1.0
    assert len(acc["per_case"]) == 2
    assert all("correct" in c for c in acc["per_case"])


# --------------------------------------------------------------------------
# run_full_eval wiring, with the judge LLM mocked out
# --------------------------------------------------------------------------


def _fake_ragas_scores(n: int) -> pd.DataFrame:
    """Stands in for score_with_ragas's returned DataFrame."""
    return pd.DataFrame(
        {
            "faithfulness": [0.9] * n,
            "context_precision": [0.8] * n,
            "context_recall": [0.75] * n,
            "answer_relevancy": [0.85] * n,
        }
    )


def test_run_full_eval_wiring_with_a_mocked_judge(monkeypatch):
    """Exercises the full collect -> score -> assemble path without a live LLM."""
    import eval.run_eval as run_eval_module

    small_testset = load_testset()[:3]  # at least one trap should be present in a real run
    monkeypatch.setattr(run_eval_module, "load_testset", lambda: small_testset)
    monkeypatch.setattr(
        run_eval_module,
        "score_with_ragas",
        lambda results: _fake_ragas_scores(len(results)),
    )

    result = run_full_eval()

    assert result["testset_size"] == 3
    assert result["good_questions"] + result["trap_questions"] == 3
    assert set(result["aggregate_metrics"].keys()) == {
        "faithfulness",
        "context_precision",
        "context_recall",
        "answer_relevancy",
    }
    assert len(result["per_question"]) == 3
    assert "verify_node_accuracy" in result


def test_report_renders_markdown_from_a_mocked_run(monkeypatch, tmp_path):
    import eval.report as report_module
    import eval.run_eval as run_eval_module

    small_testset = load_testset()[:4]
    monkeypatch.setattr(run_eval_module, "load_testset", lambda: small_testset)
    monkeypatch.setattr(
        run_eval_module,
        "score_with_ragas",
        lambda results: _fake_ragas_scores(len(results)),
    )
    monkeypatch.setattr(report_module, "RESULTS_DIR", tmp_path)

    report_path = report_module.run_and_report()

    assert report_path.exists()
    text = report_path.read_text()
    assert "## Verdict" in text
    assert "verify_node_accuracy" in text.lower()

    latest = json.loads((tmp_path / "latest.json").read_text())
    assert "overall_verdict" in latest
    assert "per_question" in latest
