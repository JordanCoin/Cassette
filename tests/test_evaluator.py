"""Tests for the dataset evaluation harness."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

import services.gateway.app as gateway_module
from libs.adapters.eval_writer import write_eval_results
from libs.adapters.jsonl_store import JsonlStore
from libs.adapters.mock_provider import MockProvider
from libs.core.contracts import DatasetRecord
from libs.core.evaluator import evaluate_records
from services.gateway.app import app


def _record(
    response: str = "A valid response here.",
    messages: list[dict[str, str]] | None = None,
    content_hash: str = "",
) -> DatasetRecord:
    return DatasetRecord(
        record_id=uuid4(),
        trace_id=uuid4(),
        source="chat",
        messages=[{"role": "user", "content": "hello"}] if messages is None else messages,
        response=response,
        model="test",
        content_hash=content_hash or uuid4().hex[:16],
    )


class TestEvaluateAccepted:
    def test_valid_record_accepted(self) -> None:
        results = evaluate_records([_record()])
        assert len(results) == 1
        assert results[0].decision == "accepted"
        assert results[0].score >= 0.7
        assert results[0].flags == []

    def test_preserves_record_id(self) -> None:
        r = _record()
        results = evaluate_records([r])
        assert results[0].record_id == r.record_id

    def test_preserves_content_hash(self) -> None:
        r = _record(content_hash="abc123")
        results = evaluate_records([r])
        assert results[0].content_hash == "abc123"


class TestEvaluateRejected:
    def test_empty_response_rejected(self) -> None:
        results = evaluate_records([_record(response="")])
        assert results[0].decision == "rejected"
        assert "empty_response" in results[0].flags

    def test_whitespace_response_rejected(self) -> None:
        results = evaluate_records([_record(response="   ")])
        assert results[0].decision == "rejected"
        assert "empty_response" in results[0].flags

    def test_no_messages_rejected(self) -> None:
        results = evaluate_records([_record(messages=[])])
        assert results[0].decision == "rejected"
        assert "no_messages" in results[0].flags

    def test_error_marker_rejected(self) -> None:
        results = evaluate_records([_record(response="[error] something broke")])
        assert results[0].decision == "rejected"
        assert any("error_marker" in f for f in results[0].flags)

    def test_traceback_marker_rejected(self) -> None:
        results = evaluate_records([_record(response="Traceback (most recent call last):")])
        assert results[0].decision == "rejected"


class TestEvaluateNeedsReview:
    def test_short_response(self) -> None:
        results = evaluate_records([_record(response="hi")])
        assert results[0].decision == "needs_review"
        assert "short_response" in results[0].flags

    def test_no_user_message(self) -> None:
        results = evaluate_records(
            [_record(messages=[{"role": "system", "content": "setup"}])]
        )
        assert results[0].decision == "needs_review"
        assert "no_user_message" in results[0].flags

    def test_mock_response(self) -> None:
        results = evaluate_records([_record(response="[mock] some response")])
        assert results[0].decision in ("rejected", "needs_review")


class TestGoldenChecks:
    def test_golden_match_accepted(self) -> None:
        r = _record(response="expected output", content_hash="gold1")
        results = evaluate_records([r], golden={"gold1": "expected output"})
        assert results[0].decision == "accepted"
        assert "golden_mismatch" not in results[0].flags

    def test_golden_mismatch_penalized(self) -> None:
        r = _record(response="wrong output", content_hash="gold1")
        results = evaluate_records([r], golden={"gold1": "expected output"})
        assert "golden_mismatch" in results[0].flags
        assert results[0].score < 1.0

    def test_no_golden_for_hash_skips_check(self) -> None:
        r = _record(content_hash="no_golden")
        results = evaluate_records([r], golden={"other": "something"})
        assert "golden_mismatch" not in results[0].flags


class TestEvalWriter:
    def test_writes_jsonl(self, tmp_path: Path) -> None:
        results = evaluate_records([_record()])
        path = tmp_path / "eval.jsonl"
        count = write_eval_results(results, path)
        assert count == 1
        line = json.loads(path.read_text().strip())
        assert line["decision"] == "accepted"

    def test_writes_multiple(self, tmp_path: Path) -> None:
        records = [_record(response=f"response {i}") for i in range(3)]
        results = evaluate_records(records)
        path = tmp_path / "eval.jsonl"
        write_eval_results(results, path)
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3


class TestEvaluateEndpoint:
    def _make_client(self, tmp_path: Path) -> TestClient:
        gateway_module.store = JsonlStore(tmp_path)
        gateway_module.provider = MockProvider()
        return TestClient(app)

    def test_evaluate_after_chat(self, tmp_path: Path) -> None:
        client = self._make_client(tmp_path)
        client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        resp = client.post("/debug/evaluate-dataset")
        assert resp.status_code == 200
        data = resp.json()
        assert data["records_evaluated"] >= 1
        assert "accepted" in data
        assert "rejected" in data
        assert "needs_review" in data

    def test_writes_eval_file(self, tmp_path: Path) -> None:
        client = self._make_client(tmp_path)
        client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        client.post("/debug/evaluate-dataset")
        assert (tmp_path / "eval_results.jsonl").exists()

    def test_empty_traces(self, tmp_path: Path) -> None:
        client = self._make_client(tmp_path)
        resp = client.post("/debug/evaluate-dataset")
        assert resp.json()["records_evaluated"] == 0
