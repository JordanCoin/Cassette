"""Tests for the orchestrator runner and stage execution."""

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

import services.gateway.app as gateway_module
from libs.adapters.jsonl_store import JsonlStore
from libs.adapters.mock_provider import MockProvider
from services.gateway.app import app
from services.orchestrator.runner import run_stage
from services.orchestrator.stages import EchoStage, FailStage


def _make_store(tmp_path: Path) -> JsonlStore:
    return JsonlStore(tmp_path)


def _make_client(tmp_path: Path) -> TestClient:
    gateway_module.store = JsonlStore(tmp_path)
    gateway_module.provider = MockProvider()
    return TestClient(app)


class TestRunnerSuccess:
    def test_echo_stage_completes(self, tmp_path: Path) -> None:
        s = _make_store(tmp_path)
        result = run_stage(EchoStage(), {"msg": "hi"}, task_store=s, event_store=s)
        assert result["status"] == "completed"
        assert result["result"] == {"echoed": {"msg": "hi"}}

    def test_creates_task(self, tmp_path: Path) -> None:
        s = _make_store(tmp_path)
        result = run_stage(EchoStage(), {}, task_store=s, event_store=s)
        task = s.get_task(result["task_id"])
        assert task is not None
        assert task.status.value == "completed"

    def test_emits_lifecycle_events(self, tmp_path: Path) -> None:
        s = _make_store(tmp_path)
        run_stage(EchoStage(), {}, task_store=s, event_store=s)
        events = s.get_latest_events(10)
        types = [e.type for e in events]
        assert "task.created" in types
        assert "stage.started" in types
        assert "stage.completed" in types

    def test_events_have_stage_name(self, tmp_path: Path) -> None:
        s = _make_store(tmp_path)
        run_stage(EchoStage(), {}, task_store=s, event_store=s)
        started = [e for e in s.get_latest_events(10) if e.type == "stage.started"]
        assert started[0].payload["stage"] == "echo"


class TestRunnerFailure:
    def test_fail_stage_returns_failed(self, tmp_path: Path) -> None:
        s = _make_store(tmp_path)
        result = run_stage(FailStage(), {}, task_store=s, event_store=s)
        assert result["status"] == "failed"
        assert "Intentional stage failure" in result["error"]

    def test_fail_stage_updates_task(self, tmp_path: Path) -> None:
        s = _make_store(tmp_path)
        result = run_stage(FailStage(), {}, task_store=s, event_store=s)
        task = s.get_task(result["task_id"])
        assert task is not None
        assert task.status.value == "failed"

    def test_fail_stage_emits_failure_event(self, tmp_path: Path) -> None:
        s = _make_store(tmp_path)
        run_stage(FailStage(), {}, task_store=s, event_store=s)
        failed_events = [e for e in s.get_latest_events(10) if e.type == "stage.failed"]
        assert len(failed_events) == 1
        assert failed_events[0].payload["error_type"] == "RuntimeError"

    def test_fail_stage_preserves_trace_id(self, tmp_path: Path) -> None:
        s = _make_store(tmp_path)
        result = run_stage(FailStage(), {}, task_store=s, event_store=s)
        task = s.get_task(result["task_id"])
        assert task is not None
        events = [e for e in s.get_latest_events(10) if e.type == "stage.failed"]
        assert str(events[0].trace_id) == str(task.trace_id)


class TestRunnerWithExistingTask:
    def test_uses_provided_task_id(self, tmp_path: Path) -> None:
        s = _make_store(tmp_path)
        from datetime import UTC, datetime
        from uuid import uuid4

        from libs.core.contracts import Task, TaskStatus

        task = Task(
            task_id=uuid4(),
            trace_id=uuid4(),
            status=TaskStatus.queued,
            created_at=datetime.now(tz=UTC),
            updated_at=datetime.now(tz=UTC),
        )
        s.save_task(task)
        result = run_stage(
            EchoStage(), {}, task_store=s, event_store=s, task_id=str(task.task_id)
        )
        assert result["task_id"] == str(task.task_id)
        assert result["status"] == "completed"


class TestOrchestratorEndpoint:
    def test_trigger_echo(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        resp = client.post("/orchestrator/run", json={"stage": "echo", "context": {"x": 1}})
        assert resp.status_code == 200
        data: dict[str, Any] = resp.json()
        assert data["status"] == "completed"
        assert data["result"]["echoed"] == {"x": 1}

    def test_default_stage_is_echo(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        resp = client.post("/orchestrator/run", json={})
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"

    def test_unknown_stage_returns_422(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        resp = client.post("/orchestrator/run", json={"stage": "nonexistent"})
        assert resp.status_code == 422

    def test_creates_task_visible_in_ledger(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        result: dict[str, Any] = client.post("/orchestrator/run", json={}).json()
        task_resp = client.get(f"/tasks/{result['task_id']}")
        assert task_resp.status_code == 200
        assert task_resp.json()["status"] == "completed"
