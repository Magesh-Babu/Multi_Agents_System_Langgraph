import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch


@pytest.fixture
def client(mock_graph):
    """TestClient with the compiled graph replaced by a mock via dependency_overrides."""
    from api.main import app
    from api.dependencies import get_graph
    app.dependency_overrides[get_graph] = lambda: mock_graph
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ── Health endpoint ───────────────────────────────────────────────────────────

def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ── /analyze endpoint ─────────────────────────────────────────────────────────

def test_analyze_returns_answer(client):
    response = client.post("/analyze", json={"query": "What is NVDA's outlook?", "thread_id": "t1"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["answer"] == "Final synthesized answer."


def test_analyze_returns_agents_used(client):
    response = client.post("/analyze", json={"query": "TSLA risk?", "thread_id": "t2"})
    data = response.json()
    assert "agents_used" in data
    assert "Technical_Analysis_Agent" in data["agents_used"]
    assert "Final_Aggregator_Agent" in data["agents_used"]


def test_analyze_returns_thread_id(client):
    response = client.post("/analyze", json={"query": "AAPL fundamentals?", "thread_id": "my-thread"})
    data = response.json()
    assert data["thread_id"] == "my-thread"


def test_analyze_default_thread_id(client):
    response = client.post("/analyze", json={"query": "MSFT sentiment?"})
    assert response.status_code == 200
    assert response.json()["thread_id"] == "default"


# ── /analyze/stream endpoint ──────────────────────────────────────────────────

def test_stream_returns_event_stream_content_type(client):
    with client.stream("POST", "/analyze/stream", json={"query": "NVDA technical?", "thread_id": "s1"}) as r:
        assert "text/event-stream" in r.headers["content-type"]


def test_stream_emits_done_event(client):
    with client.stream("POST", "/analyze/stream", json={"query": "TSLA outlook?", "thread_id": "s2"}) as r:
        lines = [line for line in r.iter_lines() if line.startswith("data:")]
    events = [line.replace("data: ", "") for line in lines]
    import json
    parsed = [json.loads(e) for e in events]
    types = [e["type"] for e in parsed]
    assert "done" in types


def test_stream_emits_final_answer_event(client):
    with client.stream("POST", "/analyze/stream", json={"query": "AMD risks?", "thread_id": "s3"}) as r:
        lines = [line for line in r.iter_lines() if line.startswith("data:")]
    import json
    parsed = [json.loads(line.replace("data: ", "")) for line in lines]
    final_events = [e for e in parsed if e["type"] == "final_answer"]
    assert len(final_events) == 1
    assert final_events[0]["content"] == "Final synthesized answer."
