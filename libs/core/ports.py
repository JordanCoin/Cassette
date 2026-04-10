"""Port interfaces for Cassette adapters.

All external I/O is behind these protocols so implementations
can be swapped per environment (file, SQLite, S3, etc.).
"""

from __future__ import annotations

from typing import Protocol

from libs.core.contracts import Event, Task, TaskStatus, Trace


class EventStore(Protocol):
    """Append-only store for events and traces."""

    def save_event(self, event: Event) -> None: ...

    def save_trace(self, trace: Trace) -> None: ...


class EventReader(Protocol):
    """Read-side interface for traces and events."""

    def get_traces_by_id(self, trace_id: str) -> list[Trace]: ...

    def get_events_by_type(self, event_type: str) -> list[Event]: ...

    def get_latest_traces(self, n: int) -> list[Trace]: ...

    def get_latest_events(self, n: int) -> list[Event]: ...


class TaskStore(Protocol):
    """Persistence for task ledger records."""

    def save_task(self, task: Task) -> None: ...

    def get_task(self, task_id: str) -> Task | None: ...

    def get_latest_tasks(self, n: int) -> list[Task]: ...

    def update_task_status(self, task_id: str, status: TaskStatus) -> Task | None: ...


class ModelProvider(Protocol):
    """Generates chat completions from a model backend."""

    @property
    def name(self) -> str: ...

    def complete(self, messages: list[dict[str, str]]) -> str: ...
