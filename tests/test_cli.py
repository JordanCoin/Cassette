"""Tests for the Cassette CLI."""

from __future__ import annotations

import json
from pathlib import Path

from libs.cli import main


def _seed_traces(data_dir: Path) -> None:
    """Write a minimal trace file so commands have data to work with."""
    from datetime import UTC, datetime
    from uuid import uuid4

    from libs.adapters.jsonl_store import JsonlStore
    from libs.core.contracts import Trace

    store = JsonlStore(data_dir)
    for i in range(3):
        store.save_trace(
            Trace(
                trace_id=uuid4(),
                timestamp=datetime.now(tz=UTC),
                messages=[{"role": "user", "content": f"msg {i}"}],
                model="test-model",
                response=f"Response number {i} with enough content.",
            )
        )


class TestCliNoArgs:
    def test_no_command_returns_1(self) -> None:
        assert main([]) == 1


class TestRunLoop:
    def test_happy_path(self, tmp_path: Path, capsys: object) -> None:
        _seed_traces(tmp_path)
        code = main(["--data-dir", str(tmp_path), "run-loop"])
        assert code == 0

    def test_empty_traces(self, tmp_path: Path) -> None:
        code = main(["--data-dir", str(tmp_path), "run-loop"])
        assert code == 0


class TestExtractDataset:
    def test_extracts(self, tmp_path: Path, capsys: object) -> None:
        import sys
        from io import StringIO

        _seed_traces(tmp_path)
        old_stdout = sys.stdout
        sys.stdout = buf = StringIO()
        try:
            code = main(["--data-dir", str(tmp_path), "extract-dataset"])
        finally:
            sys.stdout = old_stdout
        assert code == 0
        output = json.loads(buf.getvalue())
        assert output["records_extracted"] >= 1

    def test_empty(self, tmp_path: Path) -> None:
        code = main(["--data-dir", str(tmp_path), "extract-dataset"])
        assert code == 0


class TestEvaluateDataset:
    def test_evaluates(self, tmp_path: Path) -> None:
        _seed_traces(tmp_path)
        code = main(["--data-dir", str(tmp_path), "evaluate-dataset"])
        assert code == 0
        assert (tmp_path / "dataset_labeled.jsonl").exists()
        assert (tmp_path / "dataset_promoted.jsonl").exists()


class TestSnapshotDataset:
    def test_missing_promoted_returns_1(self, tmp_path: Path) -> None:
        code = main(["--data-dir", str(tmp_path), "snapshot-dataset"])
        assert code == 1

    def test_creates_snapshot(self, tmp_path: Path) -> None:
        _seed_traces(tmp_path)
        main(["--data-dir", str(tmp_path), "evaluate-dataset"])
        code = main(["--data-dir", str(tmp_path), "snapshot-dataset"])
        assert code == 0
        assert (tmp_path / "snapshots" / "manifest.jsonl").exists()


class TestListSnapshots:
    def test_empty(self, tmp_path: Path) -> None:
        code = main(["--data-dir", str(tmp_path), "list-snapshots"])
        assert code == 0

    def test_after_snapshot(self, tmp_path: Path) -> None:
        import sys
        from io import StringIO

        _seed_traces(tmp_path)
        main(["--data-dir", str(tmp_path), "evaluate-dataset"])
        main(["--data-dir", str(tmp_path), "snapshot-dataset"])

        old_stdout = sys.stdout
        sys.stdout = buf = StringIO()
        try:
            code = main(["--data-dir", str(tmp_path), "list-snapshots"])
        finally:
            sys.stdout = old_stdout
        assert code == 0
        snapshots = json.loads(buf.getvalue())
        assert len(snapshots) == 1


class TestProposeTraining:
    def test_no_snapshots_returns_1(self, tmp_path: Path) -> None:
        code = main(["--data-dir", str(tmp_path), "propose-training"])
        assert code == 1

    def test_proposes_from_latest(self, tmp_path: Path) -> None:
        _seed_traces(tmp_path)
        main(["--data-dir", str(tmp_path), "evaluate-dataset"])
        main(["--data-dir", str(tmp_path), "snapshot-dataset"])
        code = main(["--data-dir", str(tmp_path), "propose-training"])
        assert code == 0

    def test_invalid_snapshot_id_returns_1(self, tmp_path: Path) -> None:
        _seed_traces(tmp_path)
        main(["--data-dir", str(tmp_path), "evaluate-dataset"])
        main(["--data-dir", str(tmp_path), "snapshot-dataset"])
        code = main([
            "--data-dir", str(tmp_path),
            "propose-training", "--snapshot-id", "nonexistent",
        ])
        assert code == 1
