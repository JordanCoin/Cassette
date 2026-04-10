"""Tests for dataset extraction, curation, and writing."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

import services.gateway.app as gateway_module
from libs.adapters.dataset_writer import write_dataset
from libs.adapters.jsonl_store import JsonlStore
from libs.adapters.mock_provider import MockProvider
from libs.core.contracts import Trace
from libs.core.extractor import extract_records
from services.gateway.app import app


def _trace(
    model: str = "test-model",
    messages: list[dict[str, str]] | None = None,
    response: str = "test response",
) -> Trace:
    return Trace(
        trace_id=uuid4(),
        timestamp=datetime.now(tz=UTC),
        messages=[{"role": "user", "content": "hello"}] if messages is None else messages,
        model=model,
        response=response,
    )


class TestExtractRecords:
    def test_extracts_chat_trace(self) -> None:
        records = extract_records([_trace()])
        assert len(records) == 1
        assert records[0].source == "chat"
        assert records[0].model == "test-model"

    def test_skips_gateway_traces(self) -> None:
        records = extract_records([_trace(model="gateway")])
        assert records == []

    def test_skips_empty_response(self) -> None:
        records = extract_records([_trace(response="")])
        assert records == []

    def test_skips_error_response(self) -> None:
        records = extract_records([_trace(response="[error] ConnectionError: refused")])
        assert records == []

    def test_skips_empty_messages(self) -> None:
        records = extract_records([_trace(messages=[])])
        assert records == []

    def test_deduplicates(self) -> None:
        t = _trace()
        # Same content, different trace objects
        t2 = _trace(
            messages=t.messages,
            response=t.response,
            model=t.model,
        )
        records = extract_records([t, t2])
        assert len(records) == 1

    def test_different_content_not_deduped(self) -> None:
        t1 = _trace(response="response A")
        t2 = _trace(response="response B")
        records = extract_records([t1, t2])
        assert len(records) == 2

    def test_content_hash_is_set(self) -> None:
        records = extract_records([_trace()])
        assert len(records[0].content_hash) == 16

    def test_quality_defaults_to_unreviewed(self) -> None:
        records = extract_records([_trace()])
        assert records[0].quality == "unreviewed"

    def test_preserves_trace_id(self) -> None:
        t = _trace()
        records = extract_records([t])
        assert records[0].trace_id == t.trace_id


class TestDatasetWriter:
    def test_writes_jsonl(self, tmp_path: Path) -> None:
        records = extract_records([_trace()])
        path = tmp_path / "out.jsonl"
        count = write_dataset(records, path)
        assert count == 1
        assert path.exists()
        line = json.loads(path.read_text().strip())
        assert line["source"] == "chat"

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        records = extract_records([_trace()])
        path = tmp_path / "a" / "b" / "out.jsonl"
        write_dataset(records, path)
        assert path.exists()

    def test_writes_multiple(self, tmp_path: Path) -> None:
        traces = [_trace(response=f"resp {i}") for i in range(5)]
        records = extract_records(traces)
        path = tmp_path / "out.jsonl"
        count = write_dataset(records, path)
        assert count == 5
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 5

    def test_empty_records(self, tmp_path: Path) -> None:
        path = tmp_path / "out.jsonl"
        count = write_dataset([], path)
        assert count == 0


class TestExtractDatasetEndpoint:
    def _make_client(self, tmp_path: Path) -> TestClient:
        gateway_module.store = JsonlStore(tmp_path)
        gateway_module.provider = MockProvider()
        return TestClient(app)

    def test_extracts_from_chat_requests(self, tmp_path: Path) -> None:
        client = self._make_client(tmp_path)
        client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        resp = client.post("/debug/extract-dataset")
        assert resp.status_code == 200
        data = resp.json()
        assert data["records_extracted"] == 1
        # Verify file was written
        dataset_path = tmp_path / "dataset.jsonl"
        assert dataset_path.exists()

    def test_skips_gateway_only_traces(self, tmp_path: Path) -> None:
        client = self._make_client(tmp_path)
        client.get("/healthz")  # Only gateway trace
        resp = client.post("/debug/extract-dataset")
        assert resp.json()["records_extracted"] == 0

    def test_deduplicates_across_requests(self, tmp_path: Path) -> None:
        client = self._make_client(tmp_path)
        payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        client.post("/v1/chat/completions", json=payload)
        client.post("/v1/chat/completions", json=payload)
        resp = client.post("/debug/extract-dataset")
        assert resp.json()["records_extracted"] == 1
