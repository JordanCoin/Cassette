"""Tests for the training dry-run planner and plan-training stage/CLI."""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from libs.adapters.jsonl_store import JsonlStore
from libs.cli import main as cli_main
from libs.core.contracts import DatasetSnapshot
from libs.core.training_planner import build_training_plan
from services.orchestrator.runner import run_stage
from services.orchestrator.stages import PlanTrainingStage


def _make_snapshot(tmp_path: Path, records: int = 5) -> DatasetSnapshot:
    """Create a snapshot with a real dataset file."""
    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    snap_id = "20260410-120000-abc123"
    dataset_path = snapshots_dir / f"{snap_id}.jsonl"
    lines = [json.dumps({"id": i, "content": f"record {i}"}) for i in range(records)]
    dataset_path.write_text("\n".join(lines) + "\n")

    # Write manifest
    snap = DatasetSnapshot(
        snapshot_id=snap_id,
        created_at="2026-04-10T12:00:00Z",  # type: ignore[arg-type]
        content_hash="abc123",
        record_count=records,
        source_path=str(dataset_path),
    )
    manifest = snapshots_dir / "manifest.jsonl"
    manifest.write_text(snap.model_dump_json() + "\n")

    return snap


class TestBuildTrainingPlan:
    def test_valid_snapshot_produces_plan(self, tmp_path: Path) -> None:
        snap = _make_snapshot(tmp_path)
        plan = build_training_plan(snap, tmp_path)
        assert plan.snapshot_id == snap.snapshot_id
        assert plan.method in ("sft", "lora", "qlora")
        assert plan.estimated_tokens > 0
        assert plan.dataset_path != ""
        assert plan.output_dir != ""

    def test_command_is_generated(self, tmp_path: Path) -> None:
        snap = _make_snapshot(tmp_path)
        plan = build_training_plan(snap, tmp_path)
        assert "trl" in plan.command
        assert plan.base_model in plan.command

    def test_lora_command_has_peft(self, tmp_path: Path) -> None:
        snap = _make_snapshot(tmp_path, records=50)
        plan = build_training_plan(snap, tmp_path)
        assert plan.method == "lora"
        assert "--use_peft" in plan.command

    def test_qlora_command_has_4bit(self, tmp_path: Path) -> None:
        snap = _make_snapshot(tmp_path, records=200)
        plan = build_training_plan(snap, tmp_path)
        assert plan.method == "qlora"
        assert "--load_in_4bit" in plan.command

    def test_missing_dataset_raises(self, tmp_path: Path) -> None:
        snap = DatasetSnapshot(
            snapshot_id="missing",
            created_at="2026-04-10T12:00:00Z",  # type: ignore[arg-type]
            content_hash="xxx",
            record_count=5,
            source_path="/nonexistent",
        )
        with pytest.raises(ValueError, match="Dataset not found"):
            build_training_plan(snap, tmp_path)

    def test_empty_dataset_raises(self, tmp_path: Path) -> None:
        snapshots_dir = tmp_path / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        snap_id = "20260410-empty"
        (snapshots_dir / f"{snap_id}.jsonl").write_text("")
        snap = DatasetSnapshot(
            snapshot_id=snap_id,
            created_at="2026-04-10T12:00:00Z",  # type: ignore[arg-type]
            content_hash="empty",
            record_count=0,
            source_path=str(snapshots_dir / f"{snap_id}.jsonl"),
        )
        with pytest.raises(ValueError, match="empty"):
            build_training_plan(snap, tmp_path)

    def test_notes_include_count(self, tmp_path: Path) -> None:
        snap = _make_snapshot(tmp_path, records=10)
        plan = build_training_plan(snap, tmp_path)
        assert "10 records" in plan.notes


class TestPlanTrainingStage:
    def test_stage_completes(self, tmp_path: Path) -> None:
        snap = _make_snapshot(tmp_path)
        s = JsonlStore(tmp_path)
        result = run_stage(
            PlanTrainingStage(),
            {"snapshot": snap.model_dump(mode="json"), "data_dir": str(tmp_path)},
            task_store=s,
            event_store=s,
        )
        assert result["status"] == "completed"
        assert "command" in result["result"]

    def test_emits_events(self, tmp_path: Path) -> None:
        snap = _make_snapshot(tmp_path)
        s = JsonlStore(tmp_path)
        run_stage(
            PlanTrainingStage(),
            {"snapshot": snap.model_dump(mode="json"), "data_dir": str(tmp_path)},
            task_store=s,
            event_store=s,
        )
        events = s.get_latest_events(20)
        types = [e.type for e in events]
        assert "training.plan.started" in types
        assert "training.plan.completed" in types

    def test_missing_snapshot_fails(self, tmp_path: Path) -> None:
        s = JsonlStore(tmp_path)
        result = run_stage(
            PlanTrainingStage(), {}, task_store=s, event_store=s,
        )
        assert result["status"] == "failed"


class TestPlanTrainingCLI:
    def _seed(self, tmp_path: Path) -> None:
        _make_snapshot(tmp_path)

    def test_readable_output(self, tmp_path: Path) -> None:
        self._seed(tmp_path)
        old = sys.stdout
        sys.stdout = buf = StringIO()
        try:
            code = cli_main(["--data-dir", str(tmp_path), "plan-training"])
        finally:
            sys.stdout = old
        assert code == 0
        output = buf.getvalue()
        assert "Training Plan" in output
        assert "Command:" in output
        assert "trl" in output

    def test_json_output(self, tmp_path: Path) -> None:
        self._seed(tmp_path)
        old = sys.stdout
        sys.stdout = buf = StringIO()
        try:
            code = cli_main(["--data-dir", str(tmp_path), "plan-training", "--json"])
        finally:
            sys.stdout = old
        assert code == 0
        data: dict[str, Any] = json.loads(buf.getvalue())
        assert "command" in data
        assert "dataset_path" in data

    def test_no_snapshots_returns_1(self, tmp_path: Path) -> None:
        code = cli_main(["--data-dir", str(tmp_path), "plan-training"])
        assert code == 1

    def test_invalid_snapshot_id_returns_1(self, tmp_path: Path) -> None:
        self._seed(tmp_path)
        code = cli_main([
            "--data-dir", str(tmp_path),
            "plan-training", "--snapshot-id", "nonexistent",
        ])
        assert code == 1
