"""Tests for the chat completions endpoint."""

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


class TestChatCompletions:
    def test_returns_200(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        resp = client.post("/v1/chat/completions", json=VALID_PAYLOAD)
        assert resp.status_code == 200

    def test_response_shape(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        data = client.post("/v1/chat/completions", json=VALID_PAYLOAD).json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "test-model"
        assert len(data["choices"]) == 1
        choice = data["choices"][0]
        assert choice["index"] == 0
        assert choice["finish_reason"] == "stop"
        assert choice["message"]["role"] == "assistant"
        assert isinstance(choice["message"]["content"], str)

    def test_response_has_id(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        data = client.post("/v1/chat/completions", json=VALID_PAYLOAD).json()
        assert data["id"].startswith("chatcmpl-")

    def test_mock_response_is_deterministic(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        r1 = client.post("/v1/chat/completions", json=VALID_PAYLOAD).json()
        r2 = client.post("/v1/chat/completions", json=VALID_PAYLOAD).json()
        assert r1["choices"][0]["message"]["content"] == r2["choices"][0]["message"]["content"]


class TestChatValidation:
    def test_missing_model_returns_422(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 422

    def test_empty_messages_returns_422(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": []},
        )
        assert resp.status_code == 422

    def test_missing_messages_returns_422(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        resp = client.post("/v1/chat/completions", json={"model": "test"})
        assert resp.status_code == 422


class TestChatRecording:
    def test_persists_trace(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client.post("/v1/chat/completions", json=VALID_PAYLOAD)
        traces = (tmp_path / "traces.jsonl").read_text().strip().split("\n")
        assert len(traces) == 1
        record = json.loads(traces[0])
        assert record["model"] == "test-model"
        assert len(record["messages"]) == 1
        assert record["response"] == "This is a mock response."

    def test_persists_event(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client.post("/v1/chat/completions", json=VALID_PAYLOAD)
        events = (tmp_path / "events.jsonl").read_text().strip().split("\n")
        assert len(events) == 1
        record = json.loads(events[0])
        assert record["type"] == "chat.completion"
        assert record["payload"]["model"] == "test-model"
        assert record["payload"]["message_count"] == 1

    def test_trace_and_event_share_trace_id(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client.post("/v1/chat/completions", json=VALID_PAYLOAD)
        trace = json.loads((tmp_path / "traces.jsonl").read_text().strip())
        event = json.loads((tmp_path / "events.jsonl").read_text().strip())
        assert trace["trace_id"] == event["trace_id"]


class TestProviderIntegration:
    def test_custom_provider_is_used(self, tmp_path: Path) -> None:
        """Confirm the gateway routes through the provider, not hardcoded logic."""

        class StubProvider:
            @property
            def name(self) -> str:
                return "stub"

            def complete(self, messages: list[dict[str, str]], **kwargs: object) -> str:
                return "stub response"

        gateway_module.store = JsonlStore(tmp_path)
        gateway_module.provider = StubProvider()
        client = TestClient(app)
        data = client.post("/v1/chat/completions", json=VALID_PAYLOAD).json()
        assert data["choices"][0]["message"]["content"] == "stub response"

    def test_provider_name_in_event(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client.post("/v1/chat/completions", json=VALID_PAYLOAD)
        event = json.loads((tmp_path / "events.jsonl").read_text().strip())
        assert event["payload"]["provider"] == "mock"


class TestHealthzStillWorks:
    def test_healthz_after_chat(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client.post("/v1/chat/completions", json=VALID_PAYLOAD)
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
