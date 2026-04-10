"""Stage protocol and example implementations."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from libs.core.contracts import DatasetSnapshot
from libs.core.ports import WebFetch, WebSearch
from libs.core.training_executor import execute_training
from libs.core.training_plan import build_proposal
from libs.core.training_planner import build_training_plan
from libs.core.training_validator import validate_training

# Type alias for the emitter callback injected by the runner.
EmitFn = Callable[[str, dict[str, Any] | None], None]


def _noop_emit(event_type: str, payload: dict[str, Any] | None = None) -> None:
    """Default no-op emitter when no runner context is available."""


class Stage(Protocol):
    """A single unit of orchestrated work."""

    @property
    def name(self) -> str: ...

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute the stage. Returns a result dict. Raises on failure."""
        ...


class EchoStage:
    """Trivial stage that echoes its input context as the result."""

    @property
    def name(self) -> str:
        return "echo"

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        clean = {k: v for k, v in context.items() if not k.startswith("_")}
        return {"echoed": clean}


class FailStage:
    """Stage that always fails. Useful for testing error paths."""

    @property
    def name(self) -> str:
        return "fail"

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("Intentional stage failure")


class GatherSourcesStage:
    """Searches the web and fetches top results."""

    def __init__(
        self, search: WebSearch, fetch: WebFetch, max_results: int = 3
    ) -> None:
        self._search = search
        self._fetch = fetch
        self._max_results = max_results

    @property
    def name(self) -> str:
        return "gather_sources"

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        query = context.get("query", "")
        if not query:
            raise ValueError("gather_sources requires a 'query' in context")

        emit: EmitFn = context.get("_emit", _noop_emit)

        # Search
        emit("search.started", {"query": query})
        try:
            results = self._search.search(query)
        except (ConnectionError, RuntimeError) as exc:
            emit("search.failed", {"query": query, "error": str(exc)})
            raise
        top = results[: self._max_results]
        emit("search.completed", {"query": query, "result_count": len(top)})

        # Fetch
        sources: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []

        for result in top:
            url = result.get("url", "")
            emit("fetch.started", {"url": url})
            try:
                fetched = self._fetch.fetch(url)
                sources.append({
                    "title": result.get("title", ""),
                    "url": url,
                    "snippet": result.get("snippet", ""),
                    "content": fetched.get("content", ""),
                })
                emit("fetch.completed", {"url": url})
            except (ConnectionError, RuntimeError) as exc:
                errors.append({"url": url, "error": str(exc)})
                emit("fetch.failed", {"url": url, "error": str(exc)})

        return {
            "query": query,
            "source_count": len(sources),
            "sources": sources,
            "errors": errors,
        }


class ProposeTrainingStage:
    """Generates a training proposal from a dataset snapshot."""

    @property
    def name(self) -> str:
        return "propose_training"

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        emit: EmitFn = context.get("_emit", _noop_emit)

        snapshot_data = context.get("snapshot")
        if not snapshot_data:
            raise ValueError("propose_training requires 'snapshot' in context")

        snapshot = DatasetSnapshot.model_validate(snapshot_data)

        emit("proposal.started", {"snapshot_id": snapshot.snapshot_id})

        proposal = build_proposal(snapshot.snapshot_id, snapshot.record_count)

        emit("proposal.completed", {
            "snapshot_id": snapshot.snapshot_id,
            "method": proposal.method,
            "estimated_tokens": proposal.estimated_tokens,
        })

        return proposal.model_dump(mode="json")


class PlanTrainingStage:
    """Validates inputs and builds a concrete training plan."""

    @property
    def name(self) -> str:
        return "plan_training"

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        emit: EmitFn = context.get("_emit", _noop_emit)

        snapshot_data = context.get("snapshot")
        if not snapshot_data:
            raise ValueError("plan_training requires 'snapshot' in context")

        data_dir = context.get("data_dir", "data/gateway")

        snapshot = DatasetSnapshot.model_validate(snapshot_data)

        emit("training.plan.started", {"snapshot_id": snapshot.snapshot_id})

        plan = build_training_plan(snapshot, Path(data_dir))

        emit("training.plan.completed", {
            "snapshot_id": snapshot.snapshot_id,
            "method": plan.method,
            "dataset_path": plan.dataset_path,
            "output_dir": plan.output_dir,
        })

        return plan.model_dump(mode="json")


class ValidateTrainingStage:
    """Checks whether training can realistically run."""

    @property
    def name(self) -> str:
        return "validate_training"

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        emit: EmitFn = context.get("_emit", _noop_emit)

        snapshot_data = context.get("snapshot")
        if not snapshot_data:
            raise ValueError("validate_training requires 'snapshot' in context")

        data_dir = context.get("data_dir", "data/gateway")
        snapshot = DatasetSnapshot.model_validate(snapshot_data)

        emit("training.validate.started", {"snapshot_id": snapshot.snapshot_id})

        plan = build_training_plan(snapshot, Path(data_dir))
        readiness = validate_training(plan)

        emit("training.validate.completed", {
            "snapshot_id": snapshot.snapshot_id,
            "ready": readiness.ready,
            "issues": len(readiness.issues),
            "warnings": len(readiness.warnings),
        })

        return readiness.model_dump(mode="json")


class ExecuteTrainingStage:
    """Validates readiness, then executes training."""

    @property
    def name(self) -> str:
        return "execute_training"

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        emit: EmitFn = context.get("_emit", _noop_emit)

        snapshot_data = context.get("snapshot")
        if not snapshot_data:
            raise ValueError("execute_training requires 'snapshot' in context")

        data_dir = context.get("data_dir", "data/gateway")
        timeout = int(context.get("timeout", 3600))

        snapshot = DatasetSnapshot.model_validate(snapshot_data)

        # Build plan
        plan = build_training_plan(snapshot, Path(data_dir))

        # Validate first
        emit("training.validate.started", {"snapshot_id": snapshot.snapshot_id})
        readiness = validate_training(plan)
        emit("training.validate.completed", {
            "ready": readiness.ready,
            "issues": len(readiness.issues),
        })

        if not readiness.ready:
            raise RuntimeError(
                f"Training not ready: {'; '.join(readiness.issues)}"
            )

        # Execute
        emit("training.execute.started", {
            "snapshot_id": snapshot.snapshot_id,
            "method": plan.method,
            "command": plan.command[:200],
        })

        result = execute_training(plan, timeout=timeout)

        if result.success:
            emit("training.execute.completed", {
                "snapshot_id": snapshot.snapshot_id,
                "duration_sec": result.duration_sec,
                "output_dir": result.output_dir,
            })
        else:
            emit("training.execute.failed", {
                "snapshot_id": snapshot.snapshot_id,
                "exit_code": result.exit_code,
                "error": result.error[:200],
            })
            raise RuntimeError(f"Training failed: {result.error}")

        return result.model_dump(mode="json")
