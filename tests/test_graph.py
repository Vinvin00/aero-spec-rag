"""End-to-end graph behaviour over the real (small, offline) vector store."""

import pytest

from src.bounds import BY_KEY, match_query
from src.graph import build_graph, ground_spec, query_terms
from src.schemas import GroundedSpec

EXAMPLES = [
    # (query, expected quantity key, expected value, tolerance)
    ("What drag coefficient should I use for a sphere?", "drag_coefficient", 0.47, 0.01),
    ("What is the ISA air density at 10 km altitude?", "air_density", 0.4135, 0.02),
    ("What proportional navigation gain should I use for terminal homing?",
     "navigation_gain", 4.0, 1.0),
]


@pytest.fixture(scope="module")
def graph():
    return build_graph()


@pytest.mark.parametrize("query,quantity,value,tolerance", EXAMPLES)
def test_example_queries_return_grounded_values(graph, query, quantity, value, tolerance):
    result = graph.invoke({"query": query})["result"]
    spec = GroundedSpec.model_validate(result)

    assert spec.quantity == quantity
    assert spec.verified is True
    assert spec.value == pytest.approx(value, abs=tolerance)
    assert spec.unit
    assert spec.source_doc and spec.source_doc.endswith(".md")
    assert 0.0 < spec.confidence <= 1.0
    assert spec.citations, "a grounded answer must cite its retrieved passages"
    assert spec.disclaimer


@pytest.mark.parametrize("query,quantity,_v,_t", EXAMPLES)
def test_selected_value_is_inside_plausible_bounds(query, quantity, _v, _t):
    spec = GroundedSpec.model_validate(ground_spec(query))
    registry = BY_KEY[quantity]
    assert spec.plausible_bounds == [registry.low, registry.high]
    assert registry.contains(spec.value)


def test_schema_is_stable_across_examples():
    expected = set(GroundedSpec.model_fields)
    for query, *_ in EXAMPLES:
        assert set(ground_spec(query)) == expected


def test_retrieval_honours_top_k():
    spec = GroundedSpec.model_validate(ground_spec("air density at sea level", top_k=2))
    assert len(spec.citations) <= 2


def test_unmatched_quantity_degrades_gracefully():
    spec = GroundedSpec.model_validate(
        ground_spec("what is the airframe paint colour of the interceptor?")
    )
    assert spec.verified is False
    assert spec.value is None
    assert spec.confidence == 0.0
    assert any(f.code == "unknown_quantity" for f in spec.flags)


def test_out_of_bounds_value_is_flagged_and_discarded(graph):
    """A corrupted retrieval must be rejected by the verify node."""
    from src.graph import propose_node, verify_node

    poisoned = {
        "query": "drag coefficient of a sphere",
        "quantity_key": "drag_coefficient",
        "retrieved": [
            {
                "content": "| drag coefficient | 47.0 | dimensionless | corrupted row |",
                "metadata": {
                    "source_doc": "poison.md",
                    "title": "Poison",
                    "source_type": "derived",
                    "chunk_index": 0,
                },
                "score": 0.9,
            }
        ],
    }
    verified = verify_node(poisoned)
    assert verified["candidates"] == []
    assert any(f.code == "out_of_bounds" for f in verified["flags"])

    proposed = propose_node({**poisoned, **verified})
    spec = GroundedSpec.model_validate(proposed["result"])
    assert spec.verified is False
    assert spec.value is None


def test_query_terms_normalise_altitude_units():
    assert "10000" in query_terms("density at 10 km")
    assert "20000" in query_terms("intercept altitude of 20 km")


def test_query_matcher_covers_the_documented_aliases():
    assert match_query("what is the pn gain").key == "navigation_gain"
    assert match_query("typical Cd for a cube").key == "drag_coefficient"
    assert match_query("speed of sound at the tropopause").key == "speed_of_sound"
