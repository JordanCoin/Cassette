"""Tests for the end-to-end loop pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

import services.gateway.app as gateway_module
from libs.adapters.jsonl_store import JsonlStore
from libs.adapters.mock_provider import MockProvider
from libs.core.pipeline import run_full_loop
from services.gateway.app import app


def _make_store(tmp_path: Path) -> JsonlStore:
    return JsonlStore(tmp_path)


def _make_client(tmp_path: Path) -> TestClient:
    gateway_module.store = JsonlStore(tmp_path)
    gateway_module.provider = MockProvider()
    return TestClient(app)


def _seed_chat(client: TestClient, count: int = 1) -> None:
    for i in range(count):
        client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": f"msg {i}"}]},
        )


class TestFullLoopDirect:
    def test_happy_path(self, tmp_path: Path) -> None:
        s = _make_store(tmp_path)
        # Seed traces directly
        client = TestClient(app)
        gateway_module.store = s
        gateway_module.provider = MockProvider()
        _seed_chat(client, 3)

        result = run_full_loop(s, tmp_path)
        assert result["status"] == "completed"
        assert result["extracted"] >= 1
        assert "evaluation" in result
        assert result["promoted"] >= 1
        assert "snapshot" in result
        assert "proposal" in result

    def test_empty_traces(self, tmp_path: Path) -> None:
        s = _make_store(tmp_path)
        result = run_full_loop(s, tmp_path)
        assert result["status"] == "completed"
        assert result["extracted"] == 0
        assert result.get("note") == "No records to process"

    def test_has_task_id(self, tmp_path: Path) -> None:
        s = _make_store(tmp_path)
        gateway_module.store = s
        gateway_module.provider = MockProvider()
        client = TestClient(app)
        _seed_chat(client)
        result = run_full_loop(s, tmp_path)
        assert "task_id" in result
        assert "trace_id" in result

    def test_task_completed(self, tmp_path: Path) -> None:
        s = _make_store(tmp_path)
        gateway_module.store = s
        gateway_module.provider = MockProvider()
        client = TestClient(app)
        _seed_chat(client)
        result = run_full_loop(s, tmp_path)
        task = s.get_task(result["task_id"])
        assert task is not None
        assert task.status.value == "completed"

    def test_proposal_includes_method(self, tmp_path: Path) -> None:
        s = _make_store(tmp_path)
        gateway_module.store = s
        gateway_module.provider = MockProvider()
        client = TestClient(app)
        _seed_chat(client, 5)
        result = run_full_loop(s, tmp_path)
        assert result["proposal"]["method"] in ("sft", "lora", "qlora")


class TestFullLoopEvents:
    def test_emits_step_events(self, tmp_path: Path) -> None:
        s = _make_store(tmp_path)
        gateway_module.store = s
        gateway_module.provider = MockProvider()
        client = TestClient(app)
        _seed_chat(client)
        run_full_loop(s, tmp_path)

        events = s.get_latest_events(50)
        types = {e.type for e in events}
        assert "loop.extract.started" in types
        assert "loop.extract.completed" in types
        assert "loop.evaluate.completed" in types
        assert "loop.promote.completed" in types
        assert "loop.snapshot.completed" in types
        assert "loop.propose.completed" in types

    def test_all_events_share_trace_id(self, tmp_path: Path) -> None:
        s = _make_store(tmp_path)
        gateway_module.store = s
        gateway_module.provider = MockProvider()
        client = TestClient(app)
        _seed_chat(client)
        result = run_full_loop(s, tmp_path)

        loop_events = [
            e for e in s.get_latest_events(50) if e.type.startswith("loop.")
        ]
        trace_ids = {str(e.trace_id) for e in loop_events}
        assert len(trace_ids) == 1
        assert trace_ids.pop() == result["trace_id"]


class TestFullLoopEndpoint:
    def test_endpoint_happy_path(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        _seed_chat(client, 2)
        resp = client.post("/loop/run")
        assert resp.status_code == 200
        data: dict[str, Any] = resp.json()
        assert data["status"] == "completed"
        assert "proposal" in data

    def test_endpoint_empty(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        resp = client.post("/loop/run")
        data = resp.json()
        assert data["status"] == "completed"
        assert data["extracted"] == 0

    def test_endpoint_returns_snapshot(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        _seed_chat(client)
        data: dict[str, Any] = client.post("/loop/run").json()
        assert "snapshot_id" in data.get("snapshot", {})

    def test_endpoint_idempotent_snapshot(self, tmp_path: Path) -> None:
        """Running twice with same data reuses snapshot."""
        client = _make_client(tmp_path)
        _seed_chat(client)
        r1: dict[str, Any] = client.post("/loop/run").json()
        r2: dict[str, Any] = client.post("/loop/run").json()
        assert r1["status"] == "completed"
        assert r2["status"] == "completed"
