"""Tests for milestone 22: onboarding, doctor command, and CLI UX."""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path

from libs.cli import main as cli_main


def _capture_stdout(argv: list[str]) -> tuple[int, str]:
    old = sys.stdout
    sys.stdout = buf = StringIO()
    try:
        code = cli_main(argv)
    finally:
        sys.stdout = old
    return code, buf.getvalue()


def _capture_stderr(argv: list[str]) -> tuple[int, str]:
    old_err = sys.stderr
    old_out = sys.stdout
    sys.stderr = err_buf = StringIO()
    sys.stdout = StringIO()  # suppress stdout
    try:
        code = cli_main(argv)
    finally:
        sys.stderr = old_err
        sys.stdout = old_out
    return code, err_buf.getvalue()


class TestDoctorCommand:
    def test_returns_0_with_mock_provider(self, tmp_path: Path) -> None:
        code = cli_main(["--data-dir", str(tmp_path), "doctor"])
        assert code == 0

    def test_outputs_check_results(self, tmp_path: Path) -> None:
        old = sys.stdout
        sys.stdout = buf = StringIO()
        try:
            cli_main(["--data-dir", str(tmp_path), "doctor"])
        finally:
            sys.stdout = old
        output = buf.getvalue()
        assert "[OK]" in output
        assert "data_directory" in output
        assert "provider_configured" in output

    def test_reports_no_existing_data(self, tmp_path: Path) -> None:
        old = sys.stdout
        sys.stdout = buf = StringIO()
        try:
            cli_main(["--data-dir", str(tmp_path), "doctor"])
        finally:
            sys.stdout = old
        assert "No trace data" in buf.getvalue()

    def test_reports_existing_data(self, tmp_path: Path) -> None:
        # Seed some traces
        from datetime import UTC, datetime
        from uuid import uuid4

        from libs.adapters.jsonl_store import JsonlStore
        from libs.core.contracts import Trace

        store = JsonlStore(tmp_path)
        store.save_trace(Trace(
            trace_id=uuid4(), timestamp=datetime.now(tz=UTC),
            messages=[{"role": "user", "content": "hi"}],
            model="test", response="hello",
        ))

        old = sys.stdout
        sys.stdout = buf = StringIO()
        try:
            cli_main(["--data-dir", str(tmp_path), "doctor"])
        finally:
            sys.stdout = old
        assert "Trace data found" in buf.getvalue()


class TestRunLoopWithQuery:
    def test_query_seeds_and_runs(self, tmp_path: Path) -> None:
        code, output = _capture_stdout([
            "--data-dir", str(tmp_path), "run-loop", "--json", "--query", "test question",
        ])
        assert code == 0
        data = json.loads(output)
        assert data["status"] == "completed"
        assert data.get("extracted", 0) >= 1

    def test_query_without_prior_traces(self, tmp_path: Path) -> None:
        code, _ = _capture_stdout([
            "--data-dir", str(tmp_path), "run-loop", "--query", "hello world",
        ])
        assert code == 0


class TestCLIErrorMessages:
    def test_snapshot_without_promoted_shows_fix(self, tmp_path: Path) -> None:
        code, stderr = _capture_stderr([
            "--data-dir", str(tmp_path), "snapshot-dataset",
        ])
        assert code == 1
        assert "evaluate-dataset" in stderr

    def test_propose_without_snapshots_shows_fix(self, tmp_path: Path) -> None:
        code, stderr = _capture_stderr([
            "--data-dir", str(tmp_path), "propose-training",
        ])
        assert code == 1
        assert "snapshot-dataset" in stderr

    def test_evaluate_empty_shows_note(self, tmp_path: Path) -> None:
        code, output = _capture_stdout([
            "--data-dir", str(tmp_path), "evaluate-dataset",
        ])
        assert code == 0
        data = json.loads(output)
        assert data["records_evaluated"] == 0
        assert "note" in data


class TestHappyPath:
    """Validate the full user journey from zero state."""

    def test_full_onboarding_flow(self, tmp_path: Path) -> None:
        d = str(tmp_path)

        # 1. Doctor
        code = cli_main(["--data-dir", d, "doctor"])
        assert code == 0

        # 2. Run loop with query
        code, output = _capture_stdout([
            "--data-dir", d, "run-loop", "--json", "--query", "What is machine learning?",
        ])
        assert code == 0
        result = json.loads(output)
        assert result["status"] == "completed"

        # 3. List snapshots
        code, output = _capture_stdout(["--data-dir", d, "list-snapshots"])
        assert code == 0
        snapshots = json.loads(output)
        assert len(snapshots) >= 1

        # 4. Propose training
        code, output = _capture_stdout(["--data-dir", d, "propose-training"])
        assert code == 0
        proposal = json.loads(output)
        assert "method" in proposal
