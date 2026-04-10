"""End-to-end loop pipeline: extract → evaluate → promote → snapshot → propose."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from libs.adapters.dataset_writer import write_dataset
from libs.core.contracts import (
    Event,
    Task,
    TaskStatus,
)
from libs.core.evaluator import evaluate_records
from libs.core.extractor import extract_records
from libs.core.ports import EventStore
from libs.core.promoter import apply_eval_decisions, select_promoted
from libs.core.snapshots import create_snapshot
from libs.core.training_plan import build_proposal


def _emit(store: EventStore, trace_id: Any, event_type: str, payload: dict[str, Any]) -> None:
    store.save_event(
        Event(
            event_id=uuid4(),
            trace_id=trace_id,
            timestamp=datetime.now(tz=UTC),
            type=event_type,
            payload=payload,
            provenance="pipeline",
        )
    )


def run_full_loop(
    store: Any,
    data_dir: Path,
    trace_limit: int = 200,
) -> dict[str, Any]:
    """Run the full observe-to-proposal pipeline under one task."""
    now = datetime.now(tz=UTC)
    task = Task(
        task_id=uuid4(),
        trace_id=uuid4(),
        status=TaskStatus.queued,
        created_at=now,
        updated_at=now,
    )
    store.save_task(task)
    store.update_task_status(str(task.task_id), TaskStatus.running)
    tid = task.trace_id

    result: dict[str, Any] = {"task_id": str(task.task_id), "trace_id": str(tid)}

    # 1. Extract
    _emit(store, tid, "loop.extract.started", {})
    try:
        traces = store.get_latest_traces(trace_limit)
        records = extract_records(traces)
    except Exception as exc:
        return _fail(store, task, result, "extract", exc)
    result["extracted"] = len(records)
    _emit(store, tid, "loop.extract.completed", {"count": len(records)})

    if not records:
        store.update_task_status(str(task.task_id), TaskStatus.completed)
        result["status"] = "completed"
        result["note"] = "No records to process"
        return result

    # 2. Evaluate
    _emit(store, tid, "loop.evaluate.started", {})
    try:
        eval_results = evaluate_records(records)
    except Exception as exc:
        return _fail(store, task, result, "evaluate", exc)
    accepted = sum(1 for r in eval_results if r.decision == "accepted")
    rejected = sum(1 for r in eval_results if r.decision == "rejected")
    needs_review = sum(1 for r in eval_results if r.decision == "needs_review")
    result["evaluation"] = {
        "accepted": accepted,
        "rejected": rejected,
        "needs_review": needs_review,
    }
    _emit(store, tid, "loop.evaluate.completed", result["evaluation"])

    # 3. Promote
    _emit(store, tid, "loop.promote.started", {})
    try:
        labeled = apply_eval_decisions(records, eval_results)
        promoted = select_promoted(labeled)
        promoted_path = data_dir / "dataset_promoted.jsonl"
        write_dataset(promoted, promoted_path)
    except Exception as exc:
        return _fail(store, task, result, "promote", exc)
    result["promoted"] = len(promoted)
    _emit(store, tid, "loop.promote.completed", {"count": len(promoted)})

    if not promoted:
        store.update_task_status(str(task.task_id), TaskStatus.completed)
        result["status"] = "completed"
        result["note"] = "No records promoted"
        return result

    # 4. Snapshot
    _emit(store, tid, "loop.snapshot.started", {})
    try:
        snapshots_dir = data_dir / "snapshots"
        snapshot = create_snapshot(promoted_path, snapshots_dir)
    except FileExistsError:
        # Snapshot already exists for this exact content — reuse it
        from libs.core.snapshots import list_snapshots

        snapshots = list_snapshots(data_dir / "snapshots")
        snapshot = snapshots[-1] if snapshots else None  # type: ignore[assignment]
        if snapshot is None:
            return _fail(store, task, result, "snapshot", RuntimeError("No snapshot available"))
    except Exception as exc:
        return _fail(store, task, result, "snapshot", exc)
    result["snapshot"] = snapshot.model_dump(mode="json")
    _emit(store, tid, "loop.snapshot.completed", {"snapshot_id": snapshot.snapshot_id})

    # 5. Propose
    _emit(store, tid, "loop.propose.started", {})
    try:
        proposal = build_proposal(snapshot.snapshot_id, snapshot.record_count)
    except Exception as exc:
        return _fail(store, task, result, "propose", exc)
    result["proposal"] = proposal.model_dump(mode="json")
    _emit(store, tid, "loop.propose.completed", {
        "method": proposal.method,
        "snapshot_id": proposal.snapshot_id,
    })

    store.update_task_status(str(task.task_id), TaskStatus.completed)
    result["status"] = "completed"
    return result


def _fail(
    store: Any,
    task: Task,
    result: dict[str, Any],
    stage: str,
    exc: Exception,
) -> dict[str, Any]:
    store.update_task_status(str(task.task_id), TaskStatus.failed)
    _emit(store, task.trace_id, f"loop.{stage}.failed", {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    })
    result["status"] = "failed"
    result["failed_stage"] = stage
    result["error"] = str(exc)
    return result
