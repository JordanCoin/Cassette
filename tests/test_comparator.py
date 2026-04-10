"""Tests for model comparison."""

from __future__ import annotations

import json
from pathlib import Path

from libs.cli import main as cli_main
from libs.core.comparator import _score_response, compare_models


class MockModel:
    def __init__(self, name: str, response: str) -> None:
        self._name = name
        self._response = response

    @property
    def name(self) -> str:
        return self._name

    def complete(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        return self._response


class FailingModel:
    @property
    def name(self) -> str:
        return "failing"

    def complete(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        raise ConnectionError("down")


def _make_snapshot(tmp_path: Path, records: int = 5) -> Path:
    path = tmp_path / "snapshots" / "test-snap.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({
            "messages": [{"role": "user", "content": f"Review entity list {i}"}],
            "response": "original",
            "content_hash": f"hash{i}",
        })
        for i in range(records)
    ]
    path.write_text("\n".join(lines) + "\n")

    # Write manifest
    from libs.core.contracts import DatasetSnapshot

    manifest = tmp_path / "snapshots" / "manifest.jsonl"
    snap = DatasetSnapshot(
        snapshot_id="test-snap",
        created_at="2026-04-10T12:00:00Z",  # type: ignore[arg-type]
        content_hash="abc",
        record_count=records,
        source_path=str(path),
    )
    manifest.write_text(snap.model_dump_json() + "\n")
    return path


class TestScoreResponse:
    def test_valid_json_with_keys(self) -> None:
        resp = '{"keep": [{"raw_text": "FBI"}], "remove": ["junk"]}'
        assert _score_response(resp) == 1.0

    def test_valid_json_without_keys(self) -> None:
        resp = '{"entities": [1, 2, 3]}'
        assert _score_response(resp) == 0.8

    def test_code_block_penalized(self) -> None:
        resp = "```python\nprint('hello')\n```"
        assert _score_response(resp) < 0.5

    def test_empty_response(self) -> None:
        assert _score_response("") == 0.0

    def test_plain_text(self) -> None:
        assert _score_response("Some plain text answer") == 0.5


class TestCompareModels:
    def test_adapter_better(self, tmp_path: Path) -> None:
        path = _make_snapshot(tmp_path)
        base = MockModel("base", "here is some text")
        adapter = MockModel("adapter", '{"keep": [], "remove": ["junk"]}')
        result = compare_models(base, adapter, path)
        assert result.adapter_avg_score > result.base_avg_score
        assert result.improved > 0
        assert result.recommendation == "promote"

    def test_adapter_worse(self, tmp_path: Path) -> None:
        path = _make_snapshot(tmp_path)
        base = MockModel("base", '{"keep": [], "remove": []}')
        adapter = MockModel("adapter", "```python\ncode\n```")
        result = compare_models(base, adapter, path)
        assert result.base_avg_score > result.adapter_avg_score
        assert result.regressed > 0
        assert result.recommendation == "reject"

    def test_same_quality(self, tmp_path: Path) -> None:
        path = _make_snapshot(tmp_path)
        base = MockModel("base", '{"keep": []}')
        adapter = MockModel("adapter", '{"remove": []}')
        result = compare_models(base, adapter, path)
        assert result.unchanged > 0

    def test_failing_model_handled(self, tmp_path: Path) -> None:
        path = _make_snapshot(tmp_path)
        base = FailingModel()
        adapter = MockModel("adapter", '{"keep": []}')
        result = compare_models(base, adapter, path)
        assert result.records_compared > 0

    def test_missing_snapshot(self, tmp_path: Path) -> None:
        import pytest

        base = MockModel("a", "")
        adapter = MockModel("b", "")
        with pytest.raises(FileNotFoundError):
            compare_models(base, adapter, tmp_path / "nonexistent.jsonl")

    def test_details_populated(self, tmp_path: Path) -> None:
        path = _make_snapshot(tmp_path, records=3)
        base = MockModel("base", "text")
        adapter = MockModel("adapter", '{"keep": []}')
        result = compare_models(base, adapter, path)
        assert len(result.details) == 3
        assert "change" in result.details[0]


class TestCompareCLI:
    def test_no_snapshots_returns_1(self, tmp_path: Path) -> None:
        code = cli_main([
            "--data-dir", str(tmp_path),
            "compare", "--base", "a", "--adapter", "b",
        ])
        assert code == 1
