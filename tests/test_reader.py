"""Tests for the read-side data plane — JSONL store reads and debug endpoints."""

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

import services.gateway.app as gateway_module
from libs.adapters.jsonl_store import JsonlStore
from libs.adapters.mock_provider import MockProvider
from libs.core.contracts import Event, Trace
from services.gateway.app import app

VALID_PAYLOAD = {
    "model": "test-model",
    "messages": [{"role": "user", "content": "hello"}],
}


def _make_client(tmp_path: Path) -> TestClient:
    gateway_module.store = JsonlStore(tmp_path)
    gateway_module.provider = MockProvider()
    return TestClient(app)


# -- JsonlStore read tests --


class TestJsonlStoreRead:
    def _seed_store(self, tmp_path: Path) -> tuple[JsonlStore, str]:
        store = JsonlStore(tmp_path)
        tid = str(uuid4())
        now = datetime.now(tz=UTC)
        store.save_trace(
            Trace(
                trace_id=tid,  # type: ignore[arg-type]
                timestamp=now,
                messages=[{"role": "user", "content": "hi"}],
                model="test",
            )
        )
        store.save_event(
            Event(
                event_id=uuid4(),
                trace_id=tid,  # type: ignore[arg-type]
                timestamp=now,
                type="chat.completion",
                payload={"model": "test"},
            )
        )
        store.save_event(
            Event(
                event_id=uuid4(),
                trace_id=tid,  # type: ignore[arg-type]
                timestamp=now,
                type="request.completed",
                payload={"path": "/healthz"},
            )
        )
        return store, tid

    def test_get_traces_by_id(self, tmp_path: Path) -> None:
        store, tid = self._seed_store(tmp_path)
        results = store.get_traces_by_id(tid)
        assert len(results) == 1
        assert str(results[0].trace_id) == tid

    def test_get_traces_by_id_no_match(self, tmp_path: Path) -> None:
        store, _ = self._seed_store(tmp_path)
        results = store.get_traces_by_id(str(uuid4()))
        assert results == []

    def test_get_events_by_type(self, tmp_path: Path) -> None:
        store, _ = self._seed_store(tmp_path)
        results = store.get_events_by_type("chat.completion")
        assert len(results) == 1
        assert results[0].type == "chat.completion"

    def test_get_events_by_type_no_match(self, tmp_path: Path) -> None:
        store, _ = self._seed_store(tmp_path)
        results = store.get_events_by_type("nonexistent")
        assert results == []

    def test_get_latest_traces(self, tmp_path: Path) -> None:
        store, _ = self._seed_store(tmp_path)
        results = store.get_latest_traces(10)
        assert len(results) == 1

    def test_get_latest_events(self, tmp_path: Path) -> None:
        store, _ = self._seed_store(tmp_path)
        results = store.get_latest_events(10)
        assert len(results) == 2

    def test_get_latest_limits_count(self, tmp_path: Path) -> None:
        store, _ = self._seed_store(tmp_path)
        results = store.get_latest_events(1)
        assert len(results) == 1

    def test_empty_store_returns_empty(self, tmp_path: Path) -> None:
        store = JsonlStore(tmp_path)
        assert store.get_latest_traces(10) == []
        assert store.get_latest_events(10) == []


# -- Debug endpoint tests --


class TestDebugTraces:
    def test_returns_traces_after_request(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client.post("/v1/chat/completions", json=VALID_PAYLOAD)
        resp = client.get("/debug/traces")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
        assert "trace_id" in data[0]

    def test_filter_by_trace_id(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client.post("/v1/chat/completions", json=VALID_PAYLOAD)
        traces = client.get("/debug/traces").json()
        tid = traces[0]["trace_id"]
        filtered = client.get(f"/debug/traces?trace_id={tid}").json()
        assert len(filtered) == 1
        assert filtered[0]["trace_id"] == tid

    def test_filter_by_trace_id_no_match(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client.post("/v1/chat/completions", json=VALID_PAYLOAD)
        resp = client.get(f"/debug/traces?trace_id={uuid4()}")
        assert resp.json() == []

    def test_limit_param(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client.post("/v1/chat/completions", json=VALID_PAYLOAD)
        client.post("/v1/chat/completions", json=VALID_PAYLOAD)
        resp = client.get("/debug/traces?limit=1")
        assert len(resp.json()) == 1


class TestDebugEvents:
    def test_returns_events_after_request(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client.post("/v1/chat/completions", json=VALID_PAYLOAD)
        resp = client.get("/debug/events")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1

    def test_filter_by_event_type(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client.post("/v1/chat/completions", json=VALID_PAYLOAD)
        filtered = client.get("/debug/events?event_type=chat.completion").json()
        assert len(filtered) >= 1
        assert all(e["type"] == "chat.completion" for e in filtered)

    def test_filter_by_event_type_no_match(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client.post("/v1/chat/completions", json=VALID_PAYLOAD)
        resp = client.get("/debug/events?event_type=nonexistent")
        assert resp.json() == []

    def test_empty_store(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        resp = client.get("/debug/events")
        assert resp.json() == []
