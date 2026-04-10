"""Tests for the metrics registry and /metrics endpoint."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import services.gateway.app as gateway_module
from libs.adapters.jsonl_store import JsonlStore
from libs.adapters.mock_provider import MockProvider
from libs.core.metrics import registry
from services.gateway.app import app


def _make_client(tmp_path: Path) -> TestClient:
    registry.reset()
    gateway_module.store = JsonlStore(tmp_path)
    gateway_module.provider = MockProvider()
    return TestClient(app)


class TestRegistry:
    def test_inc_and_get(self) -> None:
        registry.reset()
        registry.inc("test_counter")
        assert registry.get("test_counter") == 1.0

    def test_inc_with_labels(self) -> None:
        registry.reset()
        registry.inc("test_labeled", labels={"method": "GET"})
        registry.inc("test_labeled", labels={"method": "POST"})
        assert registry.get("test_labeled", labels={"method": "GET"}) == 1.0
        assert registry.get("test_labeled", labels={"method": "POST"}) == 1.0

    def test_export_format(self) -> None:
        registry.reset()
        registry.inc("my_counter")
        registry.inc("my_counter")
        output = registry.export()
        assert "my_counter 2" in output

    def test_export_with_labels(self) -> None:
        registry.reset()
        registry.inc("req", labels={"path": "/healthz"})
        output = registry.export()
        assert 'req{path="/healthz"} 1' in output

    def test_export_empty(self) -> None:
        registry.reset()
        assert registry.export() == ""

    def test_reset(self) -> None:
        registry.reset()
        registry.inc("x")
        registry.reset()
        assert registry.get("x") == 0.0


class TestMetricsEndpoint:
    def test_returns_text_plain(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]

    def test_includes_request_counter(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client.get("/healthz")
        resp = client.get("/metrics")
        assert "cassette_requests_total" in resp.text

    def test_provider_calls_tracked(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        resp = client.get("/metrics")
        assert "cassette_provider_calls_total" in resp.text
        assert 'provider="mock"' in resp.text

    def test_task_creation_tracked(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client.post("/tasks", json={})
        resp = client.get("/metrics")
        assert "cassette_tasks_created_total" in resp.text


class TestStageMetrics:
    def test_stage_run_tracked(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client.post("/orchestrator/run", json={"stage": "echo"})
        assert registry.get("cassette_stage_runs_total", labels={"stage": "echo"}) >= 1

    def test_stage_failure_tracked(self, tmp_path: Path) -> None:
        from libs.adapters.jsonl_store import JsonlStore
        from services.orchestrator.runner import run_stage
        from services.orchestrator.stages import FailStage

        registry.reset()
        s = JsonlStore(tmp_path)
        run_stage(FailStage(), {}, task_store=s, event_store=s)
        assert registry.get("cassette_stage_failures_total", labels={"stage": "fail"}) == 1


class TestLoopMetrics:
    def test_loop_run_tracked(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        client.post("/loop/run")
        assert registry.get("cassette_loop_runs_total") >= 1
