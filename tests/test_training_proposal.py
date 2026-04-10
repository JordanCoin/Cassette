"""Tests for the training proposal stage."""

from __future__ import annotations

from pathlib import Path

import pytest

from libs.adapters.jsonl_store import JsonlStore
from services.orchestrator.runner import run_stage
from services.orchestrator.stages import ProposeTrainingStage

SNAPSHOT_SMALL = {
    "snapshot_id": "20260410-120000-abc123",
    "created_at": "2026-04-10T12:00:00Z",
    "content_hash": "abc123",
    "record_count": 5,
    "source_path": "/tmp/dataset.jsonl",
}

SNAPSHOT_MEDIUM = {
    **SNAPSHOT_SMALL,
    "snapshot_id": "20260410-120000-def456",
    "content_hash": "def456",
    "record_count": 50,
}

SNAPSHOT_LARGE = {
    **SNAPSHOT_SMALL,
    "snapshot_id": "20260410-120000-ghi789",
    "content_hash": "ghi789",
    "record_count": 200,
}


class TestProposeTrainingStage:
    def test_small_dataset_recommends_sft(self) -> None:
        stage = ProposeTrainingStage()
        result = stage.run({"snapshot": SNAPSHOT_SMALL})
        assert result["method"] == "sft"
        assert "not recommended" in result["notes"].lower()

    def test_medium_dataset_recommends_lora(self) -> None:
        stage = ProposeTrainingStage()
        result = stage.run({"snapshot": SNAPSHOT_MEDIUM})
        assert result["method"] == "lora"

    def test_large_dataset_recommends_qlora(self) -> None:
        stage = ProposeTrainingStage()
        result = stage.run({"snapshot": SNAPSHOT_LARGE})
        assert result["method"] == "qlora"

    def test_includes_snapshot_id(self) -> None:
        stage = ProposeTrainingStage()
        result = stage.run({"snapshot": SNAPSHOT_SMALL})
        assert result["snapshot_id"] == SNAPSHOT_SMALL["snapshot_id"]

    def test_includes_base_model(self) -> None:
        stage = ProposeTrainingStage()
        result = stage.run({"snapshot": SNAPSHOT_SMALL})
        assert result["base_model"] != ""

    def test_estimates_tokens(self) -> None:
        stage = ProposeTrainingStage()
        result = stage.run({"snapshot": SNAPSHOT_MEDIUM})
        assert result["estimated_tokens"] == 50 * 200

    def test_missing_snapshot_raises(self) -> None:
        stage = ProposeTrainingStage()
        with pytest.raises(ValueError, match="requires 'snapshot'"):
            stage.run({})


class TestProposeTrainingWithRunner:
    def test_completes_task(self, tmp_path: Path) -> None:
        s = JsonlStore(tmp_path)
        stage = ProposeTrainingStage()
        result = run_stage(stage, {"snapshot": SNAPSHOT_MEDIUM}, task_store=s, event_store=s)
        assert result["status"] == "completed"
        assert result["result"]["method"] == "lora"

    def test_emits_proposal_events(self, tmp_path: Path) -> None:
        s = JsonlStore(tmp_path)
        stage = ProposeTrainingStage()
        run_stage(stage, {"snapshot": SNAPSHOT_MEDIUM}, task_store=s, event_store=s)
        events = s.get_latest_events(20)
        types = [e.type for e in events]
        assert "proposal.started" in types
        assert "proposal.completed" in types

    def test_proposal_event_has_method(self, tmp_path: Path) -> None:
        s = JsonlStore(tmp_path)
        stage = ProposeTrainingStage()
        run_stage(stage, {"snapshot": SNAPSHOT_MEDIUM}, task_store=s, event_store=s)
        completed = [e for e in s.get_latest_events(20) if e.type == "proposal.completed"]
        assert completed[0].payload["method"] == "lora"
        assert completed[0].payload["snapshot_id"] == SNAPSHOT_MEDIUM["snapshot_id"]

    def test_failure_on_missing_snapshot(self, tmp_path: Path) -> None:
        s = JsonlStore(tmp_path)
        stage = ProposeTrainingStage()
        result = run_stage(stage, {}, task_store=s, event_store=s)
        assert result["status"] == "failed"

    def test_all_events_share_trace_id(self, tmp_path: Path) -> None:
        s = JsonlStore(tmp_path)
        stage = ProposeTrainingStage()
        run_stage(stage, {"snapshot": SNAPSHOT_SMALL}, task_store=s, event_store=s)
        events = s.get_latest_events(20)
        trace_ids = {str(e.trace_id) for e in events}
        assert len(trace_ids) == 1
