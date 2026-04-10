"""Tests for the demo command and loop output formatting."""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

from libs.cli import _format_loop_result
from libs.cli import main as cli_main


class TestFormatLoopResult:
    def test_completed_result(self) -> None:
        result = {
            "status": "completed",
            "extracted": 5,
            "evaluation": {"accepted": 4, "rejected": 1, "needs_review": 0},
            "promoted": 4,
            "snapshot": {"snapshot_id": "20260410-abc123"},
            "proposal": {
                "method": "lora",
                "base_model": "meta-llama/Llama-3.1-8B-Instruct",
                "estimated_tokens": 800,
            },
        }
        output = _format_loop_result(result)
        assert "completed" in output
        assert "5 records" in output
        assert "4 accepted" in output
        assert "1 rejected" in output
        assert "4 records" in output
        assert "20260410-abc123" in output
        assert "lora" in output
        assert "800 tokens" in output

    def test_empty_result(self) -> None:
        result = {"status": "completed", "extracted": 0, "note": "No records to process"}
        output = _format_loop_result(result)
        assert "No records" in output

    def test_failed_result(self) -> None:
        result: dict[str, object] = {"status": "failed", "failed_stage": "extract", "error": "boom"}
        output = _format_loop_result(result)
        assert "failed" in output
        assert "extract" in output
        assert "boom" in output


class TestRunLoopOutput:
    def test_default_output_is_readable(self, tmp_path: Path) -> None:
        from libs.adapters.jsonl_store import JsonlStore
        from libs.core.recorder import record_chat

        store = JsonlStore(tmp_path)
        record_chat(
            store, model="test", messages=[{"role": "user", "content": "hi"}],
            response="hello there friend", latency_ms=0.0, provider="mock",
        )

        old = sys.stdout
        sys.stdout = buf = StringIO()
        try:
            cli_main(["--data-dir", str(tmp_path), "run-loop"])
        finally:
            sys.stdout = old
        output = buf.getvalue()
        # Should be readable text, not raw JSON
        assert "Status:" in output
        assert "Extracted:" in output

    def test_json_flag_outputs_json(self, tmp_path: Path) -> None:
        import json

        from libs.adapters.jsonl_store import JsonlStore
        from libs.core.recorder import record_chat

        store = JsonlStore(tmp_path)
        record_chat(
            store, model="test", messages=[{"role": "user", "content": "hi"}],
            response="hello there friend", latency_ms=0.0, provider="mock",
        )

        old = sys.stdout
        sys.stdout = buf = StringIO()
        try:
            cli_main(["--data-dir", str(tmp_path), "run-loop", "--json"])
        finally:
            sys.stdout = old
        data = json.loads(buf.getvalue())
        assert data["status"] == "completed"


class TestDemoCommand:
    def test_demo_runs(self, tmp_path: Path) -> None:
        old = sys.stdout
        sys.stdout = buf = StringIO()
        try:
            code = cli_main(["--data-dir", str(tmp_path), "demo"])
        finally:
            sys.stdout = old
        assert code == 0
        output = buf.getvalue()
        assert "Demo" in output
        assert "Seeded:" in output
        assert "Artifacts" in output or "results" in output

    def test_demo_creates_artifacts(self, tmp_path: Path) -> None:
        old = sys.stdout
        sys.stdout = StringIO()
        try:
            cli_main(["--data-dir", str(tmp_path), "demo"])
        finally:
            sys.stdout = old
        assert (tmp_path / "traces.jsonl").exists()
        assert (tmp_path / "dataset_promoted.jsonl").exists()
        assert (tmp_path / "snapshots").exists()

    def test_demo_shows_next_steps(self, tmp_path: Path) -> None:
        old = sys.stdout
        sys.stdout = buf = StringIO()
        try:
            cli_main(["--data-dir", str(tmp_path), "demo"])
        finally:
            sys.stdout = old
        output = buf.getvalue()
        assert "list-snapshots" in output
        assert "propose-training" in output
