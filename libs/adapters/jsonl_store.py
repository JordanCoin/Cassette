"""JSONL file-based event store.

Append-only persistence to local `.jsonl` files.
One file per record type (events.jsonl, traces.jsonl).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from libs.core.contracts import Event, Task, TaskStatus, Trace


class JsonlStore:
    """Persists and reads events and traces as newline-delimited JSON."""

    def __init__(self, directory: Path) -> None:
        self._dir = directory
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def events_path(self) -> Path:
        return self._dir / "events.jsonl"

    @property
    def traces_path(self) -> Path:
        return self._dir / "traces.jsonl"

    @property
    def tasks_path(self) -> Path:
        return self._dir / "tasks.jsonl"

    # -- write --

    def save_event(self, event: Event) -> None:
        with open(self.events_path, "a") as f:
            f.write(event.model_dump_json() + "\n")

    def save_trace(self, trace: Trace) -> None:
        with open(self.traces_path, "a") as f:
            f.write(trace.model_dump_json() + "\n")

    # -- read --

    def _read_lines(self, path: Path) -> list[dict[str, object]]:
        if not path.exists():
            return []
        lines = path.read_text().strip().split("\n")
        return [json.loads(line) for line in lines if line]

    def get_traces_by_id(self, trace_id: str) -> list[Trace]:
        return [
            Trace.model_validate(row)
            for row in self._read_lines(self.traces_path)
            if row.get("trace_id") == trace_id
        ]

    def get_events_by_type(self, event_type: str) -> list[Event]:
        return [
            Event.model_validate(row)
            for row in self._read_lines(self.events_path)
            if row.get("type") == event_type
        ]

    def get_latest_traces(self, n: int) -> list[Trace]:
        rows = self._read_lines(self.traces_path)
        return [Trace.model_validate(row) for row in rows[-n:]]

    def get_latest_events(self, n: int) -> list[Event]:
        rows = self._read_lines(self.events_path)
        return [Event.model_validate(row) for row in rows[-n:]]

    # -- tasks --

    def save_task(self, task: Task) -> None:
        with open(self.tasks_path, "a") as f:
            f.write(task.model_dump_json() + "\n")

    def get_task(self, task_id: str) -> Task | None:
        for row in reversed(self._read_lines(self.tasks_path)):
            if row.get("task_id") == task_id:
                return Task.model_validate(row)
        return None

    def get_latest_tasks(self, n: int) -> list[Task]:
        rows = self._read_lines(self.tasks_path)
        return [Task.model_validate(row) for row in rows[-n:]]

    def update_task_status(self, task_id: str, status: TaskStatus) -> Task | None:
        rows = self._read_lines(self.tasks_path)
        found = False
        for row in rows:
            if row.get("task_id") == task_id:
                row["status"] = status.value
                row["updated_at"] = datetime.now(tz=UTC).isoformat()
                found = True
        if not found:
            return None
        with open(self.tasks_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return self.get_task(task_id)
