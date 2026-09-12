"""HTTP contract for the FastAPI layer."""

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.schemas import GroundedSpec


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["collection"]


def test_ground_spec_returns_valid_schema(client):
    response = client.post(
        "/ground-spec", json={"query": "What drag coefficient should I use for a sphere?"}
    )
    assert response.status_code == 200
    spec = GroundedSpec.model_validate(response.json())
    assert spec.quantity == "drag_coefficient"
    assert spec.verified is True
    assert spec.value == pytest.approx(0.47, abs=0.01)
    assert spec.source_doc
    assert spec.citations


@pytest.mark.parametrize(
    "query",
    [
        "ISA air density at 10 km",
        "proportional navigation gain range",
        "speed of sound at sea level",
    ],
)
def test_ground_spec_across_queries(client, query):
    response = client.post("/ground-spec", json={"query": query})
    assert response.status_code == 200
    spec = GroundedSpec.model_validate(response.json())
    assert spec.verified is True
    assert spec.value is not None
    assert spec.disclaimer


def test_top_k_is_forwarded(client):
    response = client.post("/ground-spec", json={"query": "air density", "top_k": 1})
    assert response.status_code == 200
    assert len(response.json()["citations"]) <= 1


def test_empty_query_is_rejected(client):
    assert client.post("/ground-spec", json={"query": ""}).status_code == 422
    assert client.post("/ground-spec", json={}).status_code == 422
