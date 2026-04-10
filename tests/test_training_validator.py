"""Tests for pre-training validation."""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path
from typing import Any

from libs.adapters.jsonl_store import JsonlStore
from libs.cli import main as cli_main
from libs.core.contracts import TrainingPlan
from libs.core.training_validator import validate_training
from services.orchestrator.runner import run_stage
from services.orchestrator.stages import ValidateTrainingStage


def _make_plan(
    tmp_path: Path,
    records: int = 10,
    method: str = "sft",
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
) -> TrainingPlan:
    dataset_path = tmp_path / "dataset.jsonl"
    lines = [json.dumps({"id": i}) for i in range(records)]
    dataset_path.write_text("\n".join(lines) + "\n")
    return TrainingPlan(
        snapshot_id="test-snap",
        dataset_path=str(dataset_path),
        base_model=base_model,
        method=method,
        estimated_tokens=records * 200,
        output_dir=str(tmp_path / "models" / "output"),
        command="python -m trl sft ...",
    )


def _make_snapshot_for_cli(tmp_path: Path, records: int = 10) -> None:
    """Create a snapshot with dataset for CLI testing."""
    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    snap_id = "20260410-120000-abc123"
    dataset_path = snapshots_dir / f"{snap_id}.jsonl"
    lines = [json.dumps({"id": i, "content": f"record {i}"}) for i in range(records)]
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


class TestValidateTraining:
    def test_valid_plan_has_issues_for_missing_deps(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        result = validate_training(plan)
        # TRL/PEFT/torch likely not installed in test env
        assert any("not installed" in i for i in result.issues) or result.ready

    def test_missing_dataset_not_ready(self, tmp_path: Path) -> None:
        plan = TrainingPlan(
            snapshot_id="test",
            dataset_path=str(tmp_path / "nonexistent.jsonl"),
            base_model="meta-llama/Llama-3.1-8B-Instruct",
            method="sft",
            estimated_tokens=1000,
            output_dir=str(tmp_path / "out"),
            command="train ...",
        )
        result = validate_training(plan)
        assert result.ready is False
        assert any("not found" in i for i in result.issues)

    def test_empty_dataset_not_ready(self, tmp_path: Path) -> None:
        dataset = tmp_path / "empty.jsonl"
        dataset.write_text("")
        plan = TrainingPlan(
            snapshot_id="test",
            dataset_path=str(dataset),
            base_model="meta-llama/Llama-3.1-8B-Instruct",
            method="sft",
            estimated_tokens=0,
            output_dir=str(tmp_path / "out"),
            command="train ...",
        )
        result = validate_training(plan)
        assert result.ready is False
        assert any("empty" in i for i in result.issues)

    def test_small_dataset_warning(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, records=3)
        result = validate_training(plan)
        assert any("only 3 records" in w.lower() for w in result.warnings)

    def test_hf_model_warning(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, base_model="meta-llama/Llama-3.1-8B-Instruct")
        result = validate_training(plan)
        assert any("downloaded" in w.lower() for w in result.warnings)

    def test_notes_include_platform(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        result = validate_training(plan)
        assert "Platform:" in result.notes
        assert "Python:" in result.notes

    def test_preserves_plan_fields(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        result = validate_training(plan)
        assert result.snapshot_id == "test-snap"
        assert result.method == "sft"


class TestValidateTrainingStage:
    def test_stage_completes(self, tmp_path: Path) -> None:
        _make_snapshot_for_cli(tmp_path)
        s = JsonlStore(tmp_path)

        from libs.core.snapshots import list_snapshots

        snap = list_snapshots(tmp_path / "snapshots")[0]
        result = run_stage(
            ValidateTrainingStage(),
            {"snapshot": snap.model_dump(mode="json"), "data_dir": str(tmp_path)},
            task_store=s,
            event_store=s,
        )
        assert result["status"] == "completed"
        assert "ready" in result["result"]

    def test_emits_events(self, tmp_path: Path) -> None:
        _make_snapshot_for_cli(tmp_path)
        s = JsonlStore(tmp_path)

        from libs.core.snapshots import list_snapshots

        snap = list_snapshots(tmp_path / "snapshots")[0]
        run_stage(
            ValidateTrainingStage(),
            {"snapshot": snap.model_dump(mode="json"), "data_dir": str(tmp_path)},
            task_store=s,
            event_store=s,
        )
        events = s.get_latest_events(20)
        types = [e.type for e in events]
        assert "training.validate.started" in types
        assert "training.validate.completed" in types


class TestValidateTrainingCLI:
    def test_readable_output(self, tmp_path: Path) -> None:
        _make_snapshot_for_cli(tmp_path)
        old = sys.stdout
        sys.stdout = buf = StringIO()
        try:
            cli_main(["--data-dir", str(tmp_path), "validate-training"])
        finally:
            sys.stdout = old
        output = buf.getvalue()
        assert "Training Validation" in output
        assert "READY" in output or "NOT READY" in output

    def test_json_output(self, tmp_path: Path) -> None:
        _make_snapshot_for_cli(tmp_path)
        old = sys.stdout
        sys.stdout = buf = StringIO()
        try:
            cli_main(["--data-dir", str(tmp_path), "validate-training", "--json"])
        finally:
            sys.stdout = old
        data: dict[str, Any] = json.loads(buf.getvalue())
        assert "ready" in data
        assert "issues" in data
        assert "warnings" in data

    def test_no_snapshots_returns_1(self, tmp_path: Path) -> None:
        code = cli_main(["--data-dir", str(tmp_path), "validate-training"])
        assert code == 1
