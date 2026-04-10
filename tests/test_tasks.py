"""Tests for the task ledger API."""

from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi.testclient import TestClient

import services.gateway.app as gateway_module
from libs.adapters.jsonl_store import JsonlStore
from libs.adapters.mock_provider import MockProvider
from services.gateway.app import app


def _make_client(tmp_path: Path) -> TestClient:
    gateway_module.store = JsonlStore(tmp_path)
    gateway_module.provider = MockProvider()
    return TestClient(app)


def _create_task(client: TestClient) -> dict[str, Any]:
    resp = client.post("/tasks", json={})
    assert resp.status_code == 201
    data: dict[str, Any] = resp.json()
    return data


class TestCreateTask:
    def test_returns_201(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        resp = client.post("/tasks", json={})
        assert resp.status_code == 201

    def test_returns_task_with_defaults(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        data = _create_task(client)
        assert data["status"] == "queued"
        assert "task_id" in data
        assert "trace_id" in data
        assert data["parent_task_id"] is None

    def test_accepts_trace_id(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        tid = str(uuid4())
        data = client.post("/tasks", json={"trace_id": tid}).json()
        assert data["trace_id"] == tid

    def test_accepts_parent_task_id(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        pid = str(uuid4())
        data = client.post("/tasks", json={"parent_task_id": pid}).json()
        assert data["parent_task_id"] == pid


class TestListTasks:
    def test_empty_list(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        resp = client.get("/tasks")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_created_tasks(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        _create_task(client)
        _create_task(client)
        data = client.get("/tasks").json()
        assert len(data) == 2

    def test_limit_param(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        for _ in range(3):
            _create_task(client)
        data = client.get("/tasks?limit=2").json()
        assert len(data) == 2


class TestGetTask:
    def test_returns_task(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        created = _create_task(client)
        resp = client.get(f"/tasks/{created['task_id']}")
        assert resp.status_code == 200
        assert resp.json()["task_id"] == created["task_id"]

    def test_not_found(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        resp = client.get(f"/tasks/{uuid4()}")
        assert resp.status_code == 404


class TestUpdateTask:
    def test_update_status(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        created = _create_task(client)
        resp = client.patch(
            f"/tasks/{created['task_id']}",
            json={"status": "running"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "running"

    def test_update_to_completed(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        created = _create_task(client)
        client.patch(f"/tasks/{created['task_id']}", json={"status": "running"})
        resp = client.patch(f"/tasks/{created['task_id']}", json={"status": "completed"})
        assert resp.json()["status"] == "completed"

    def test_invalid_status_returns_422(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        created = _create_task(client)
        resp = client.patch(
            f"/tasks/{created['task_id']}",
            json={"status": "invalid"},
        )
        assert resp.status_code == 422

    def test_not_found_returns_404(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        resp = client.patch(f"/tasks/{uuid4()}", json={"status": "running"})
        assert resp.status_code == 404

    def test_updated_at_changes(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        created = _create_task(client)
        resp = client.patch(
            f"/tasks/{created['task_id']}",
            json={"status": "running"},
        )
        assert resp.json()["updated_at"] != created["updated_at"]


class TestTaskEvents:
    def test_create_emits_event(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        _create_task(client)
        events = client.get("/debug/events?event_type=task.created").json()
        assert len(events) == 1
        assert events[0]["payload"]["status"] == "queued"

    def test_update_emits_event(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        created = _create_task(client)
        client.patch(f"/tasks/{created['task_id']}", json={"status": "running"})
        events = client.get("/debug/events?event_type=task.updated").json()
        assert len(events) == 1
        assert events[0]["payload"]["status"] == "running"

    def test_event_links_trace_id(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        created = _create_task(client)
        events = client.get("/debug/events?event_type=task.created").json()
        assert events[0]["trace_id"] == created["trace_id"]
