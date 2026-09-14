"""Run the eval testset through the real pipeline and score it with RAGAS.

    python -m eval.run_eval

This module is split into two halves on purpose:

* `collect_results` runs every testset question through the actual
  `src.graph` pipeline (never mocked) and needs no judge LLM at all -- it's
  the fast, free, fully deterministic half. `tests/test_eval_harness.py`
  exercises this half directly.
* `score_with_ragas` sends those collected results to a judge LLM via RAGAS
  for the four metrics. This is the slow, non-free (or Ollama-slow) half, and
  only `eval/report.py`'s manual run is expected to call it.

The two are kept apart so "does the harness work" (tests, CI-safe) and "how
good is the pipeline" (a real eval run) are never accidentally conflated.

IMPORTANT: `ground_spec()`/the pipeline is called with the testset question
only. `ground_truth_answer` is never passed into retrieval or generation --
it is used exclusively afterwards, to score what the pipeline already
produced. See DECISIONS.md if you're checking this isn't leaking.
"""

from __future__ import annotations

# Must precede any `ragas` import anywhere in this process -- see the module
# docstring in _ragas_compat.py for why.
from . import _ragas_compat  # noqa: F401  (import for side effect)

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.graph import get_graph

from .testset import load_testset

# Ollama by default: no API key, consistent with the rest of this repo (see
# src/llm.py). qwen2.5:3b, not the larger models the rest of this project
# uses, is the default judge specifically -- see DECISIONS.md: RAGAS's
# metrics need structured JSON output at every step, and in practice a small
# instruction-tuned model complied far more reliably than a larger
# general-purpose one that kept failing ragas's output parser. It is also
# light enough to run in a memory-constrained Ollama instance. Override with
# AERO_EVAL_LLM_MODEL / AERO_EVAL_LLM_BACKEND for a different judge.
EVAL_LLM_BACKEND = os.environ.get("AERO_EVAL_LLM_BACKEND", "ollama")
EVAL_LLM_MODEL = os.environ.get("AERO_EVAL_LLM_MODEL", "qwen2.5:3b")
EVAL_OLLAMA_BASE_URL = os.environ.get("AERO_EVAL_OLLAMA_BASE_URL", "http://localhost:11434")
EVAL_EMBED_MODEL = os.environ.get(
    "AERO_EVAL_EMBED_MODEL", "hf.co/CompendiumLabs/bge-small-en-v1.5-gguf"
)

# Matches propose_node's fixed answer template in src/graph.py:
# f"{key} = {value} {unit}{' (' + notes + ')' if notes else ''}, from {doc}."
_CITATION_CLAUSE = re.compile(r",\s*from\s+\S+\s*\.?\s*$")


def _strip_citation_clause(answer: str) -> str:
    """Drop the trailing ", from <file>." for faithfulness scoring only.

    See the comment at its call site in score_with_ragas for why: confirmed
    against the judge's own per-statement output that this clause is always
    scored unsupported, for a reason that has nothing to do with whether the
    pipeline's answer is correct.
    """
    return _CITATION_CLAUSE.sub("", answer).strip()


@dataclass
class PipelineResult:
    """One testset question run through the real pipeline."""

    question: str
    ground_truth_answer: str
    expected_source_doc: Optional[str]
    is_trap: bool
    expected_verified: bool
    # Full (untruncated) retrieved chunk text -- read from the pipeline's own
    # internal state, not from GroundedSpec.citations[].snippet, which the API
    # truncates to 300 chars for response-payload size, not for evaluation.
    retrieved_contexts: List[str] = field(default_factory=list)
    # Narrower context for faithfulness only -- see collect_results.
    faithfulness_context: List[str] = field(default_factory=list)
    generated_answer: str = ""
    verified: bool = False
    source_doc: Optional[str] = None
    confidence: float = 0.0


def collect_results(testset: Optional[List[dict]] = None) -> List[PipelineResult]:
    """Run every testset case through the real pipeline. No LLM judge involved."""
    cases = testset if testset is not None else load_testset()
    graph = get_graph()
    results: List[PipelineResult] = []
    for case in cases:
        # Only the question reaches the pipeline -- see module docstring.
        state = graph.invoke({"query": case["question"]})
        result = state["result"]
        retrieved_contexts = [r["content"] for r in state.get("retrieved", [])]

        # A retrieved chunk is a whole markdown table (several rows, e.g. one
        # per body shape or altitude); the pipeline only actually used one row
        # of it to answer. That's the right context for *faithfulness*
        # ("is the answer supported by what was actually used") -- handing a
        # judge the whole table instead makes correct answers look weakly
        # supported (which of eight values is this?). It is deliberately the
        # *wrong* context for context_precision/context_recall, which score
        # retrieval quality itself and need the real, full retrieved set to
        # mean anything -- narrowing those too would make them trivially
        # perfect and hide real retrieval gaps. See score_with_ragas.
        selected = state.get("selected")
        if selected is not None:
            faithfulness_context = [
                f"| Quantity | Value | Unit | Notes |\n| --- | --- | --- | --- |\n"
                f"| {selected.label} | {selected.value_repr()} | {selected.unit} | "
                f"{selected.notes} |"
            ]
        else:
            faithfulness_context = retrieved_contexts

        results.append(
            PipelineResult(
                question=case["question"],
                ground_truth_answer=case["ground_truth_answer"],
                expected_source_doc=case["expected_source_doc"],
                is_trap=case["is_trap"],
                expected_verified=case["expected_verified"],
                retrieved_contexts=retrieved_contexts,
                faithfulness_context=faithfulness_context,
                generated_answer=result.get("answer", ""),
                verified=result.get("verified", False),
                source_doc=result.get("source_doc"),
                confidence=result.get("confidence", 0.0),
            )
        )
    return results


def verify_node_accuracy(results: List[PipelineResult]) -> Dict[str, Any]:
    """Custom (non-RAGAS) check: did `verified` match expectation per case?

    True for the 12 "good" questions (a real value should be found and pass
    bounds-checking); False for the 3 trap questions (no value exists in the
    corpus, so the correct behavior is to decline, not fabricate).
    """
    per_case = [
        {
            "question": r.question,
            "is_trap": r.is_trap,
            "expected_verified": r.expected_verified,
            "actual_verified": r.verified,
            "correct": r.verified == r.expected_verified,
        }
        for r in results
    ]
    correct = sum(1 for c in per_case if c["correct"])
    return {
        "accuracy": correct / len(per_case) if per_case else 0.0,
        "correct": correct,
        "total": len(per_case),
        "per_case": per_case,
    }


def _build_ragas_judge():
    """Wrap a chat model + embeddings for RAGAS. Ollama by default, no API key."""
    from langchain_ollama import OllamaEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    if EVAL_LLM_BACKEND == "ollama":
        from langchain_ollama import ChatOllama

        chat = ChatOllama(model=EVAL_LLM_MODEL, base_url=EVAL_OLLAMA_BASE_URL, temperature=0.0)
    elif EVAL_LLM_BACKEND == "anthropic":
        from langchain_anthropic import ChatAnthropic

        chat = ChatAnthropic(model=EVAL_LLM_MODEL, temperature=0.0)
    else:
        raise ValueError(f"Unknown AERO_EVAL_LLM_BACKEND: {EVAL_LLM_BACKEND!r}")

    # answer_relevancy scores embedding similarity between the answer and a
    # synthetic question, so it needs a real paraphrase-aware embedder --
    # this project's own default (offline hashing) is explicitly weak at
    # that and was tanking the score. Same Ollama model this repo already
    # uses elsewhere for AERO_EMBEDDINGS=ollama.
    embed = OllamaEmbeddings(model=EVAL_EMBED_MODEL, base_url=EVAL_OLLAMA_BASE_URL)
    return LangchainLLMWrapper(chat), LangchainEmbeddingsWrapper(embed)


def score_with_ragas(results: List[PipelineResult]):
    """Score every collected result with the four required RAGAS metrics.

    Computed over the full testset, traps included, per the task brief ("for
    each testset entry ... compute with ragas"). Traps have no valid
    answer/context by construction, so their per-metric scores are expected
    to be low -- that's a correct outcome, not a bug, and the report does not
    hold trap rows to the pass thresholds (see DECISIONS.md). Correctness for
    traps is judged by `verify_node_accuracy` instead.
    """
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    from ragas.run_config import RunConfig

    llm, embeddings = _build_ragas_judge()
    # A local Ollama instance serves one generation at a time regardless of
    # how many concurrent jobs ragas fires, so max_workers>1 only means many
    # jobs sit queued behind each other until ragas's own per-job timeout
    # (default 180s) elapses -- observed directly: 7 of 8 jobs timed out at
    # the default settings on a 2-question smoke test. max_workers=1 makes
    # the queueing honest instead of hidden, and 600s gives a small local
    # model room to finish its (often multi-call) chain per job.
    run_config = RunConfig(timeout=600, max_workers=1)

    def _run(metrics, context_field, strip_citation=False):
        dataset = EvaluationDataset(
            samples=[
                SingleTurnSample(
                    user_input=r.question,
                    retrieved_contexts=getattr(r, context_field) or [""],
                    response=(
                        _strip_citation_clause(r.generated_answer)
                        if strip_citation
                        else r.generated_answer
                    ),
                    reference=r.ground_truth_answer,
                )
                for r in results
            ]
        )
        return evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            show_progress=False,
            run_config=run_config,
        ).to_pandas()

    # faithfulness asks "is the answer supported by what was actually used to
    # produce it" -- retrieved_contexts (the whole table chunk) is the wrong
    # scope for that; faithfulness_context (the one row actually cited) is
    # right. context_precision/context_recall score retrieval itself and need
    # the real, full retrieved set -- see collect_results for the full
    # reasoning on why these two must NOT share a context field.
    #
    # strip_citation=True here only: confirmed directly (not guessed) by
    # inspecting the judge's own per-statement verdicts -- every answer's
    # trailing ", from <file>." is extracted as its own claim and always
    # fails, reason verbatim "The context does not provide any information
    # about the source of the data." True: a bare table row can never
    # confirm which file it came from: that's retrieval-time metadata, not
    # a fact the row's text asserts about itself. Penalizing faithfulness
    # for a citation the row was never going to contain isn't measuring
    # anything -- the citation's correctness is exactly what
    # `expected_source_doc` / `verify_node_accuracy` already check.
    faithfulness_df = _run([faithfulness], "faithfulness_context", strip_citation=True)
    other_df = _run([context_precision, context_recall, answer_relevancy], "retrieved_contexts")
    other_df["faithfulness"] = faithfulness_df["faithfulness"]
    return other_df


def run_full_eval() -> Dict[str, Any]:
    """Collect + score. Returns one combined results dict."""
    results = collect_results()
    verify_acc = verify_node_accuracy(results)
    ragas_df = score_with_ragas(results)

    per_question = []
    for i, r in enumerate(results):
        row = ragas_df.iloc[i]
        per_question.append(
            {
                "question": r.question,
                "is_trap": r.is_trap,
                "expected_source_doc": r.expected_source_doc,
                "source_doc": r.source_doc,
                "expected_verified": r.expected_verified,
                "verified": r.verified,
                "confidence": r.confidence,
                "generated_answer": r.generated_answer,
                "ground_truth_answer": r.ground_truth_answer,
                "faithfulness": float(row["faithfulness"]),
                "context_precision": float(row["context_precision"]),
                "context_recall": float(row["context_recall"]),
                "answer_relevancy": float(row["answer_relevancy"]),
            }
        )

    good_rows = [q for q in per_question if not q["is_trap"]]

    def _mean(key: str, rows) -> float:
        values = [r[key] for r in rows if r[key] == r[key]]  # drop NaN
        return sum(values) / len(values) if values else 0.0

    metrics = ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]
    aggregate = {m: _mean(m, good_rows) for m in metrics}
    aggregate_all = {m: _mean(m, per_question) for m in metrics}

    return {
        "aggregate_metrics": aggregate,
        "aggregate_metrics_including_traps": aggregate_all,
        "verify_node_accuracy": verify_acc,
        "per_question": per_question,
        "judge_llm": f"{EVAL_LLM_BACKEND}:{EVAL_LLM_MODEL}",
        "testset_size": len(results),
        "good_questions": len(good_rows),
        "trap_questions": len(per_question) - len(good_rows),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_full_eval(), indent=2, default=str))
