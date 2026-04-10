"""Tests for the gateway service."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

import services.gateway.app as gateway_module
from libs.adapters.jsonl_store import JsonlStore
from services.gateway.app import app


class TestHealthz:
    def test_healthz_returns_ok(self) -> None:
        client = TestClient(app)
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_healthz_content_type(self) -> None:
        client = TestClient(app)
        response = client.get("/healthz")
        assert "application/json" in response.headers["content-type"]


class TestRecording:
    def _make_client(self, tmp_path: Path) -> TestClient:
        """Swap the gateway store to a temp directory for test isolation."""
        gateway_module.store = JsonlStore(tmp_path)
        return TestClient(app)

    def test_request_persists_trace(self, tmp_path: Path) -> None:
        client = self._make_client(tmp_path)
        client.get("/healthz")
        traces_file = tmp_path / "traces.jsonl"
        assert traces_file.exists()
        lines = traces_file.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert "trace_id" in record
        assert record["model"] == "gateway"

    def test_request_persists_event(self, tmp_path: Path) -> None:
        client = self._make_client(tmp_path)
        client.get("/healthz")
        events_file = tmp_path / "events.jsonl"
        assert events_file.exists()
        lines = events_file.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["type"] == "request.completed"
        assert record["payload"]["path"] == "/healthz"
        assert record["payload"]["status_code"] == 200
        assert record["provenance"] == "gateway"

    def test_trace_and_event_share_trace_id(self, tmp_path: Path) -> None:
        client = self._make_client(tmp_path)
        client.get("/healthz")
        trace = json.loads((tmp_path / "traces.jsonl").read_text().strip())
        event = json.loads((tmp_path / "events.jsonl").read_text().strip())
        assert trace["trace_id"] == event["trace_id"]

    def test_multiple_requests_append(self, tmp_path: Path) -> None:
        client = self._make_client(tmp_path)
        client.get("/healthz")
        client.get("/healthz")
        traces = (tmp_path / "traces.jsonl").read_text().strip().split("\n")
        events = (tmp_path / "events.jsonl").read_text().strip().split("\n")
        assert len(traces) == 2
        assert len(events) == 2
