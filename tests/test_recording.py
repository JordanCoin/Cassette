"""Tests for enriched trace/event recording — success and failure paths."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

import services.gateway.app as gateway_module
from libs.adapters.jsonl_store import JsonlStore
from libs.adapters.mock_provider import MockProvider
from services.gateway.app import app

VALID_PAYLOAD = {
    "model": "test-model",
    "messages": [{"role": "user", "content": "hello"}],
}


def _make_client(tmp_path: Path) -> TestClient:
    gateway_module.store = JsonlStore(tmp_path)
    gateway_module.provider = MockProvider()
    return TestClient(app)


class TestSuccessMetadata:
    def test_event_has_status_ok(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client.post("/v1/chat/completions", json=VALID_PAYLOAD)
        event = json.loads((tmp_path / "events.jsonl").read_text().strip())
        assert event["payload"]["status"] == "ok"

    def test_event_has_provider(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client.post("/v1/chat/completions", json=VALID_PAYLOAD)
        event = json.loads((tmp_path / "events.jsonl").read_text().strip())
        assert event["payload"]["provider"] == "mock"

    def test_event_has_backend_url(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client.post("/v1/chat/completions", json=VALID_PAYLOAD)
        event = json.loads((tmp_path / "events.jsonl").read_text().strip())
        assert "backend_url" in event["payload"]

    def test_event_has_latency(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client.post("/v1/chat/completions", json=VALID_PAYLOAD)
        event = json.loads((tmp_path / "events.jsonl").read_text().strip())
        assert isinstance(event["payload"]["latency_ms"], float)


class TestFailureRecording:
    def _make_failing_client(self, tmp_path: Path, exc: Exception) -> TestClient:
        class FailingProvider:
            @property
            def name(self) -> str:
                return "failing"

            def complete(self, messages: list[dict[str, str]]) -> str:
                raise exc

        gateway_module.store = JsonlStore(tmp_path)
        gateway_module.provider = FailingProvider()
        return TestClient(app)

    def test_connection_error_returns_502(self, tmp_path: Path) -> None:
        client = self._make_failing_client(tmp_path, ConnectionError("refused"))
        resp = client.post("/v1/chat/completions", json=VALID_PAYLOAD)
        assert resp.status_code == 502

    def test_connection_error_response_shape(self, tmp_path: Path) -> None:
        client = self._make_failing_client(tmp_path, ConnectionError("refused"))
        data = client.post("/v1/chat/completions", json=VALID_PAYLOAD).json()
        assert data["error"]["type"] == "ConnectionError"
        assert "refused" in data["error"]["message"]

    def test_runtime_error_returns_502(self, tmp_path: Path) -> None:
        client = self._make_failing_client(tmp_path, RuntimeError("bad response"))
        resp = client.post("/v1/chat/completions", json=VALID_PAYLOAD)
        assert resp.status_code == 502

    def test_failure_persists_error_trace(self, tmp_path: Path) -> None:
        client = self._make_failing_client(tmp_path, ConnectionError("refused"))
        client.post("/v1/chat/completions", json=VALID_PAYLOAD)
        trace = json.loads((tmp_path / "traces.jsonl").read_text().strip())
        assert "[error]" in trace["response"]
        assert "ConnectionError" in trace["response"]

    def test_failure_persists_error_event(self, tmp_path: Path) -> None:
        client = self._make_failing_client(tmp_path, ConnectionError("refused"))
        client.post("/v1/chat/completions", json=VALID_PAYLOAD)
        event = json.loads((tmp_path / "events.jsonl").read_text().strip())
        assert event["type"] == "chat.completion.error"
        assert event["payload"]["status"] == "error"
        assert event["payload"]["error_type"] == "ConnectionError"
        assert event["payload"]["provider"] == "failing"

    def test_failure_event_has_latency(self, tmp_path: Path) -> None:
        client = self._make_failing_client(tmp_path, RuntimeError("timeout"))
        client.post("/v1/chat/completions", json=VALID_PAYLOAD)
        event = json.loads((tmp_path / "events.jsonl").read_text().strip())
        assert isinstance(event["payload"]["latency_ms"], float)

    def test_failure_trace_and_event_share_trace_id(self, tmp_path: Path) -> None:
        client = self._make_failing_client(tmp_path, ConnectionError("refused"))
        client.post("/v1/chat/completions", json=VALID_PAYLOAD)
        trace = json.loads((tmp_path / "traces.jsonl").read_text().strip())
        event = json.loads((tmp_path / "events.jsonl").read_text().strip())
        assert trace["trace_id"] == event["trace_id"]

    def test_error_message_is_truncated(self, tmp_path: Path) -> None:
        long_msg = "x" * 5000
        client = self._make_failing_client(tmp_path, RuntimeError(long_msg))
        client.post("/v1/chat/completions", json=VALID_PAYLOAD)
        event = json.loads((tmp_path / "events.jsonl").read_text().strip())
        assert len(event["payload"]["error_message"]) < 5000
        assert event["payload"]["error_message"].endswith("...[truncated]")
