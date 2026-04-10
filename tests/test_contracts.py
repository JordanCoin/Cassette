"""Tests for canonical contract models."""

from datetime import UTC, datetime
from uuid import uuid4

from libs.core.contracts import Event, Task, TaskStatus, Trace


class TestEvent:
    def test_create_event(self) -> None:
        event = Event(
            event_id=uuid4(),
            trace_id=uuid4(),
            timestamp=datetime.now(tz=UTC),
            type="trace.started",
            payload={"model": "test"},
        )
        assert event.type == "trace.started"
        assert event.provenance == ""

    def test_event_schema_exportable(self) -> None:
        schema = Event.model_json_schema()
        assert "properties" in schema
        assert "event_id" in schema["properties"]


class TestTask:
    def test_create_task(self) -> None:
        now = datetime.now(tz=UTC)
        task = Task(
            task_id=uuid4(),
            trace_id=uuid4(),
            status=TaskStatus.queued,
            created_at=now,
            updated_at=now,
        )
        assert task.status == TaskStatus.queued
        assert task.parent_task_id is None

    def test_task_status_transitions(self) -> None:
        now = datetime.now(tz=UTC)
        task = Task(
            task_id=uuid4(),
            trace_id=uuid4(),
            status=TaskStatus.queued,
            created_at=now,
            updated_at=now,
        )
        updated = task.model_copy(update={"status": TaskStatus.running})
        assert updated.status == TaskStatus.running

    def test_task_schema_exportable(self) -> None:
        schema = Task.model_json_schema()
        assert "properties" in schema
        assert "status" in schema["properties"]


class TestTrace:
    def test_create_trace(self) -> None:
        trace = Trace(
            trace_id=uuid4(),
            timestamp=datetime.now(tz=UTC),
            messages=[{"role": "user", "content": "hello"}],
            model="test-model",
        )
        assert trace.model == "test-model"
        assert trace.tool_calls == []
        assert trace.response == ""

    def test_trace_schema_exportable(self) -> None:
        schema = Trace.model_json_schema()
        assert "properties" in schema
        assert "trace_id" in schema["properties"]
