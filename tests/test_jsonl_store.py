"""Tests for JSONL file-based event store."""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from libs.adapters.jsonl_store import JsonlStore
from libs.core.contracts import Event, Trace


class TestJsonlStore:
    def test_save_event_creates_file(self, tmp_path: Path) -> None:
        store = JsonlStore(tmp_path)
        event = Event(
            event_id=uuid4(),
            trace_id=uuid4(),
            timestamp=datetime.now(tz=UTC),
            type="test.event",
            payload={"key": "value"},
        )
        store.save_event(event)
        assert store.events_path.exists()

    def test_save_event_writes_valid_json(self, tmp_path: Path) -> None:
        store = JsonlStore(tmp_path)
        event = Event(
            event_id=uuid4(),
            trace_id=uuid4(),
            timestamp=datetime.now(tz=UTC),
            type="test.event",
            payload={"key": "value"},
        )
        store.save_event(event)
        line = store.events_path.read_text().strip()
        parsed = json.loads(line)
        assert parsed["type"] == "test.event"
        assert parsed["payload"] == {"key": "value"}

    def test_save_trace_creates_file(self, tmp_path: Path) -> None:
        store = JsonlStore(tmp_path)
        trace = Trace(
            trace_id=uuid4(),
            timestamp=datetime.now(tz=UTC),
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
        )
        store.save_trace(trace)
        assert store.traces_path.exists()

    def test_save_trace_writes_valid_json(self, tmp_path: Path) -> None:
        store = JsonlStore(tmp_path)
        trace = Trace(
            trace_id=uuid4(),
            timestamp=datetime.now(tz=UTC),
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
        )
        store.save_trace(trace)
        line = store.traces_path.read_text().strip()
        parsed = json.loads(line)
        assert parsed["model"] == "test-model"
        assert parsed["messages"] == [{"role": "user", "content": "test"}]

    def test_multiple_appends(self, tmp_path: Path) -> None:
        store = JsonlStore(tmp_path)
        for i in range(3):
            event = Event(
                event_id=uuid4(),
                trace_id=uuid4(),
                timestamp=datetime.now(tz=UTC),
                type=f"test.event.{i}",
                payload={},
            )
            store.save_event(event)
        lines = store.events_path.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_creates_directory_if_missing(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b"
        store = JsonlStore(nested)
        assert nested.exists()
        event = Event(
            event_id=uuid4(),
            trace_id=uuid4(),
            timestamp=datetime.now(tz=UTC),
            type="test",
            payload={},
        )
        store.save_event(event)
        assert store.events_path.exists()
