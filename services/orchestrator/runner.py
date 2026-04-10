"""Orchestrator runner — executes a stage within a task lifecycle."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from libs.core.contracts import Event, Task, TaskStatus
from libs.core.ports import EventStore, TaskStore
from services.orchestrator.stages import Stage


def _emit(
    store: EventStore, task: Task, event_type: str, extra: dict[str, Any] | None = None
) -> None:
    payload: dict[str, Any] = {
        "task_id": str(task.task_id),
        "status": task.status.value,
    }
    if extra:
        payload.update(extra)
    store.save_event(
        Event(
            event_id=uuid4(),
            trace_id=task.trace_id,
            timestamp=datetime.now(tz=UTC),
            type=event_type,
            payload=payload,
            provenance="orchestrator",
        )
    )


def run_stage(
    stage: Stage,
    context: dict[str, Any],
    task_store: TaskStore,
    event_store: EventStore,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Execute a single stage, managing the full task lifecycle.

    If task_id is provided, uses an existing task. Otherwise creates one.
    Returns a result dict with task_id, status, and stage output or error.
    """
    now = datetime.now(tz=UTC)

    # Create or fetch task
    if task_id:
        task = task_store.get_task(task_id)
        if task is None:
            raise ValueError(f"Task not found: {task_id}")
    else:
        task = Task(
            task_id=uuid4(),
            trace_id=uuid4(),
            status=TaskStatus.queued,
            created_at=now,
            updated_at=now,
        )
        task_store.save_task(task)
        _emit(event_store, task, "task.created")

    # Mark running
    updated = task_store.update_task_status(str(task.task_id), TaskStatus.running)
    if updated:
        task = updated
    _emit(event_store, task, "stage.started", {"stage": stage.name})

    # Inject emitter for stages to use for step-level events
    def stage_emit(event_type: str, payload: dict[str, Any] | None = None) -> None:
        _emit(event_store, task, event_type, payload)

    context = {**context, "_emit": stage_emit}

    # Execute
    try:
        result = stage.run(context)
    except Exception as exc:
        task_store.update_task_status(str(task.task_id), TaskStatus.failed)
        _emit(event_store, task, "stage.failed", {
            "stage": stage.name,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        })
        return {
            "task_id": str(task.task_id),
            "status": "failed",
            "error": str(exc),
        }

    # Mark completed
    task_store.update_task_status(str(task.task_id), TaskStatus.completed)
    _emit(event_store, task, "stage.completed", {"stage": stage.name})

    return {
        "task_id": str(task.task_id),
        "status": "completed",
        "result": result,
    }
