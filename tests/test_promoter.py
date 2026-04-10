"""Tests for dataset promotion logic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi.testclient import TestClient

import services.gateway.app as gateway_module
from libs.adapters.jsonl_store import JsonlStore
from libs.adapters.mock_provider import MockProvider
from libs.core.contracts import DatasetRecord, EvalResult
from libs.core.promoter import apply_eval_decisions, select_promoted
from services.gateway.app import app


def _record(quality: str = "unreviewed", record_id: str | None = None) -> DatasetRecord:
    return DatasetRecord(
        record_id=record_id or uuid4(),  # type: ignore[arg-type]
        trace_id=uuid4(),
        source="chat",
        messages=[{"role": "user", "content": "hello"}],
        response="A valid response.",
        model="test",
        content_hash=uuid4().hex[:16],
        quality=quality,
    )


def _make_eval_result(record: DatasetRecord, decision: str) -> EvalResult:
    return EvalResult(
        record_id=record.record_id,
        content_hash=record.content_hash,
        decision=decision,
        flags=[],
        score=1.0 if decision == "accepted" else 0.5,
    )


class TestApplyEvalDecisions:
    def test_updates_quality(self) -> None:
        r = _record()
        e = _make_eval_result(r, "accepted")
        labeled = apply_eval_decisions([r], [e])
        assert labeled[0].quality == "accepted"

    def test_does_not_mutate_original(self) -> None:
        r = _record()
        e = _make_eval_result(r, "accepted")
        apply_eval_decisions([r], [e])
        assert r.quality == "unreviewed"

    def test_rejected_decision(self) -> None:
        r = _record()
        e = _make_eval_result(r, "rejected")
        labeled = apply_eval_decisions([r], [e])
        assert labeled[0].quality == "rejected"

    def test_needs_review_decision(self) -> None:
        r = _record()
        e = _make_eval_result(r, "needs_review")
        labeled = apply_eval_decisions([r], [e])
        assert labeled[0].quality == "needs_review"

    def test_unmatched_record_unchanged(self) -> None:
        r = _record()
        labeled = apply_eval_decisions([r], [])
        assert labeled[0].quality == "unreviewed"

    def test_multiple_records(self) -> None:
        r1 = _record()
        r2 = _record()
        e1 = _make_eval_result(r1, "accepted")
        e2 = _make_eval_result(r2, "rejected")
        labeled = apply_eval_decisions([r1, r2], [e1, e2])
        assert labeled[0].quality == "accepted"
        assert labeled[1].quality == "rejected"


class TestSelectPromoted:
    def test_returns_only_accepted(self) -> None:
        records = [
            _record(quality="accepted"),
            _record(quality="rejected"),
            _record(quality="needs_review"),
            _record(quality="accepted"),
        ]
        promoted = select_promoted(records)
        assert len(promoted) == 2
        assert all(r.quality == "accepted" for r in promoted)

    def test_empty_when_none_accepted(self) -> None:
        records = [_record(quality="rejected"), _record(quality="needs_review")]
        assert select_promoted(records) == []

    def test_empty_input(self) -> None:
        assert select_promoted([]) == []


class TestPromoteEndpoint:
    def _make_client(self, tmp_path: Path) -> TestClient:
        gateway_module.store = JsonlStore(tmp_path)
        gateway_module.provider = MockProvider()
        return TestClient(app)

    def test_promote_after_chat(self, tmp_path: Path) -> None:
        client = self._make_client(tmp_path)
        client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        resp = client.post("/debug/promote-dataset")
        assert resp.status_code == 200
        data: dict[str, Any] = resp.json()
        assert data["total_records"] >= 1

    def test_writes_labeled_file(self, tmp_path: Path) -> None:
        client = self._make_client(tmp_path)
        client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        client.post("/debug/promote-dataset")
        labeled_path = tmp_path / "dataset_labeled.jsonl"
        assert labeled_path.exists()
        record = json.loads(labeled_path.read_text().strip().split("\n")[0])
        assert record["quality"] != "unreviewed"

    def test_writes_promoted_file(self, tmp_path: Path) -> None:
        client = self._make_client(tmp_path)
        client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        client.post("/debug/promote-dataset")
        promoted_path = tmp_path / "dataset_promoted.jsonl"
        assert promoted_path.exists()

    def test_empty_traces(self, tmp_path: Path) -> None:
        client = self._make_client(tmp_path)
        resp = client.post("/debug/promote-dataset")
        assert resp.json()["total_records"] == 0
        assert resp.json()["promoted"] == 0
