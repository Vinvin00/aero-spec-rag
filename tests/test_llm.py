"""LLM-backed behaviour: the safety guards, and a live Ollama smoke test.

The guard tests stub the model, so they run offline and are the ones that matter
for CI. The live tests are skipped unless an Ollama server is actually serving
the configured model.
"""

import json
import urllib.request

import pytest

from src import config, llm
from src.bounds import QUANTITIES
from src.schemas import GroundedSpec


# --------------------------------------------------------------------------
# stubbed-model guard tests (always run)
# --------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeModel:
    def __init__(self, content):
        self._content = content
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        return FakeResponse(self._content)


class ExplodingModel:
    def invoke(self, prompt):
        raise RuntimeError("ollama is down")


@pytest.fixture
def stub_llm(monkeypatch):
    def _install(model):
        monkeypatch.setattr(llm, "get_llm", lambda: model)
        return model

    return _install


KEYS = [q.key for q in QUANTITIES]


def test_classify_accepts_only_registry_keys(stub_llm):
    stub_llm(FakeModel("launch_mass"))
    assert llm.classify_quantity("how heavy is it", KEYS) == "launch_mass"


def test_classify_rejects_invented_key(stub_llm):
    """A hallucinated label must not enter the pipeline."""
    stub_llm(FakeModel("warhead_yield"))
    assert llm.classify_quantity("how big is the bang", KEYS) is None


def test_classify_handles_none_and_noise(stub_llm):
    stub_llm(FakeModel("NONE"))
    assert llm.classify_quantity("what colour is it", KEYS) is None
    stub_llm(FakeModel("`air_density`."))
    assert llm.classify_quantity("density", KEYS) == "air_density"


def test_classify_survives_backend_failure(stub_llm):
    stub_llm(ExplodingModel())
    assert llm.classify_quantity("anything", KEYS) is None


def test_narrate_appends_citation_deterministically(stub_llm):
    stub_llm(FakeModel("The launch mass is 700 kg"))
    sentence = llm.narrate("launch_mass", "700", "kg", "medium-range", "interceptor.md")
    assert sentence == "The launch mass is 700 kg. Source: interceptor.md."


def test_narrate_rejects_a_sentence_that_drops_the_value(stub_llm):
    """The verified number must survive verbatim, or the prose is discarded."""
    stub_llm(FakeModel("The launch mass is roughly three quarters of a tonne."))
    assert llm.narrate("launch_mass", "700", "kg", "", "interceptor.md") is None


def test_narrate_rejects_a_altered_value(stub_llm):
    stub_llm(FakeModel("The launch mass is 750 kg."))
    assert llm.narrate("launch_mass", "700", "kg", "", "interceptor.md") is None


def test_narrate_survives_backend_failure(stub_llm):
    stub_llm(ExplodingModel())
    assert llm.narrate("launch_mass", "700", "kg", "", "interceptor.md") is None


def test_disabled_backend_is_a_no_op(monkeypatch):
    monkeypatch.setattr(config, "LLM_BACKEND", "none")
    llm.get_llm.cache_clear()
    assert llm.llm_enabled() is False
    assert llm.get_llm() is None
    assert llm.narrate("q", "1", "m", "", "d.md") is None
    assert llm.classify_quantity("q", KEYS) is None
    llm.get_llm.cache_clear()


def test_graph_narration_falls_back_when_model_rejects(monkeypatch):
    """A rejected narration keeps the deterministic answer and raises a flag."""
    from src import graph

    monkeypatch.setattr(graph, "llm_enabled", lambda: True)
    monkeypatch.setattr(graph, "narrate", lambda **kwargs: None)

    spec = GroundedSpec.model_validate(
        graph.ground_spec("What drag coefficient should I use for a sphere?")
    )
    assert spec.verified is True
    assert spec.value == pytest.approx(0.47, abs=0.01)
    assert spec.narration_model is None
    assert any(f.code == "narration_unavailable" for f in spec.flags)


def test_graph_narration_is_applied_when_accepted(monkeypatch):
    from src import graph

    monkeypatch.setattr(graph, "llm_enabled", lambda: True)
    monkeypatch.setattr(
        graph, "narrate", lambda **kwargs: "A sphere has a Cd of 0.47. Source: x.md."
    )

    spec = GroundedSpec.model_validate(
        graph.ground_spec("What drag coefficient should I use for a sphere?")
    )
    assert spec.answer == "A sphere has a Cd of 0.47. Source: x.md."
    assert spec.narration_model.startswith("none:") or spec.narration_model
    # The structured fields are unchanged by narration.
    assert spec.value == pytest.approx(0.47, abs=0.01)
    assert spec.source_doc.endswith(".md")


# --------------------------------------------------------------------------
# live Ollama tests (skipped when the server or model is absent)
# --------------------------------------------------------------------------


def _ollama_models():
    try:
        with urllib.request.urlopen(
            f"{config.OLLAMA_BASE_URL}/api/tags", timeout=2
        ) as response:
            return {m["name"] for m in json.load(response).get("models", [])}
    except Exception:
        return set()


AVAILABLE = _ollama_models()
needs_chat = pytest.mark.skipif(
    config.LLM_MODEL not in AVAILABLE,
    reason=f"ollama model {config.LLM_MODEL!r} not available locally",
)
needs_embed = pytest.mark.skipif(
    config.OLLAMA_EMBED_MODEL not in {n.split(":")[0] for n in AVAILABLE},
    reason=f"ollama model {config.OLLAMA_EMBED_MODEL!r} not pulled",
)


@needs_chat
def test_live_ollama_classifies_a_paraphrased_query(monkeypatch):
    monkeypatch.setattr(config, "LLM_BACKEND", "ollama")
    llm.get_llm.cache_clear()
    try:
        key = llm.classify_quantity("how heavy is the bird at launch?", KEYS)
        assert key in set(KEYS) or key is None
    finally:
        llm.get_llm.cache_clear()


@needs_chat
def test_live_ollama_narration_preserves_the_value(monkeypatch):
    monkeypatch.setattr(config, "LLM_BACKEND", "ollama")
    llm.get_llm.cache_clear()
    try:
        sentence = llm.narrate(
            "air_density", "0.4135", "kg/m^3", "ISA at 10000 m altitude", "isa.md"
        )
        if sentence is not None:
            assert "0.4135" in sentence
            assert sentence.endswith("Source: isa.md.")
    finally:
        llm.get_llm.cache_clear()


@needs_embed
def test_live_ollama_embeddings_have_stable_dimension(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_BACKEND", "ollama")
    from src.embeddings import get_embeddings

    embedder = get_embeddings()
    vectors = embedder.embed_documents(["drag coefficient of a sphere", "ISA density"])
    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1]) > 0
