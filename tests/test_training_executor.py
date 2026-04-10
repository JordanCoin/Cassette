"""Tests for training execution."""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path

from libs.adapters.jsonl_store import JsonlStore
from libs.cli import main as cli_main
from libs.core.contracts import TrainingPlan
from libs.core.training_executor import execute_training
from services.orchestrator.runner import run_stage
from services.orchestrator.stages import ExecuteTrainingStage


def _make_plan(tmp_path: Path) -> TrainingPlan:
    dataset = tmp_path / "dataset.jsonl"
    lines = [json.dumps({"id": i}) for i in range(10)]
    dataset.write_text("\n".join(lines) + "\n")
    return TrainingPlan(
        snapshot_id="test-snap",
        dataset_path=str(dataset),
        base_model="test-model",
        method="sft",
        estimated_tokens=2000,
        output_dir=str(tmp_path / "output"),
        command="echo training_complete",
    )


def _make_snapshot_for_cli(tmp_path: Path, records: int = 10) -> None:
    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    snap_id = "20260410-120000-abc123"
    dataset_path = snapshots_dir / f"{snap_id}.jsonl"
    lines = [json.dumps({"id": i}) for i in range(records)]
    dataset_path.write_text("\n".join(lines) + "\n")

    from libs.core.contracts import DatasetSnapshot

    snap = DatasetSnapshot(
        snapshot_id=snap_id,
        created_at="2026-04-10T12:00:00Z",  # type: ignore[arg-type]
        content_hash="abc123",
        record_count=records,
        source_path=str(dataset_path),
    )
    manifest = snapshots_dir / "manifest.jsonl"
    manifest.write_text(snap.model_dump_json() + "\n")


class TestExecuteTraining:
    def test_successful_command(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        result = execute_training(plan)
        assert result.success is True
        assert result.exit_code == 0
        assert result.duration_sec >= 0
        assert "training_complete" in result.stdout_tail

    def test_failed_command(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        plan = plan.model_copy(update={"command": "false"})  # exit code 1
        result = execute_training(plan)
        assert result.success is False
        assert result.exit_code != 0

    def test_command_not_found(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        plan = plan.model_copy(update={"command": "nonexistent_training_binary_xyz"})
        result = execute_training(plan)
        assert result.success is False
        assert result.exit_code == -1
        assert "not found" in result.error.lower()

    def test_timeout(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        plan = plan.model_copy(update={"command": "sleep 10"})
        result = execute_training(plan, timeout=1)
        assert result.success is False
        assert result.exit_code == -2
        assert "timed out" in result.error.lower()

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        out_dir = tmp_path / "new" / "output"
        plan = plan.model_copy(update={"output_dir": str(out_dir)})
        execute_training(plan)
        assert out_dir.exists()

    def test_captures_stdout(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        plan = plan.model_copy(update={"command": "echo line1 && echo line2"})
        result = execute_training(plan)
        assert "line1" in result.stdout_tail
        assert "line2" in result.stdout_tail


class TestExecuteTrainingStage:
    def test_stage_fails_when_not_ready(self, tmp_path: Path) -> None:
        """Training stage should fail if validate_training reports not ready."""
        _make_snapshot_for_cli(tmp_path)
        s = JsonlStore(tmp_path)

        from libs.core.snapshots import list_snapshots

        snap = list_snapshots(tmp_path / "snapshots")[0]
        result = run_stage(
            ExecuteTrainingStage(),
            {"snapshot": snap.model_dump(mode="json"), "data_dir": str(tmp_path)},
            task_store=s,
            event_store=s,
        )
        # Will fail because TRL/torch not installed
        assert result["status"] == "failed"

    def test_emits_validate_events(self, tmp_path: Path) -> None:
        _make_snapshot_for_cli(tmp_path)
        s = JsonlStore(tmp_path)

        from libs.core.snapshots import list_snapshots

        snap = list_snapshots(tmp_path / "snapshots")[0]
        run_stage(
            ExecuteTrainingStage(),
            {"snapshot": snap.model_dump(mode="json"), "data_dir": str(tmp_path)},
            task_store=s,
            event_store=s,
        )
        events = s.get_latest_events(20)
        types = [e.type for e in events]
        assert "training.validate.started" in types


class TestTrainCLI:
    def test_no_snapshots_returns_1(self, tmp_path: Path) -> None:
        code = cli_main(["--data-dir", str(tmp_path), "train"])
        assert code == 1

    def test_with_snapshot_attempts_training(self, tmp_path: Path) -> None:
        _make_snapshot_for_cli(tmp_path)
        old = sys.stdout
        sys.stdout = buf = StringIO()
        try:
            code = cli_main(["--data-dir", str(tmp_path), "train"])
        finally:
            sys.stdout = old
        # Will fail validation (no TRL) but should not crash
        assert code == 1
        assert "FAILED" in buf.getvalue() or "not installed" in buf.getvalue().lower()
