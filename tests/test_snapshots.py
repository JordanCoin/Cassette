"""Tests for dataset snapshot management."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import services.gateway.app as gateway_module
from libs.adapters.jsonl_store import JsonlStore
from libs.adapters.mock_provider import MockProvider
from libs.core.snapshots import create_snapshot, list_snapshots
from services.gateway.app import app


def _make_dataset(tmp_path: Path, lines: int = 3) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "dataset_promoted.jsonl"
    content = "\n".join(f'{{"id": {i}}}' for i in range(lines)) + "\n"
    path.write_text(content)
    return path


class TestCreateSnapshot:
    def test_creates_snapshot_file(self, tmp_path: Path) -> None:
        dataset = _make_dataset(tmp_path)
        snap_dir = tmp_path / "snapshots"
        snapshot = create_snapshot(dataset, snap_dir)
        dest = snap_dir / f"{snapshot.snapshot_id}.jsonl"
        assert dest.exists()

    def test_snapshot_metadata(self, tmp_path: Path) -> None:
        dataset = _make_dataset(tmp_path, lines=5)
        snap_dir = tmp_path / "snapshots"
        snapshot = create_snapshot(dataset, snap_dir)
        assert snapshot.record_count == 5
        assert len(snapshot.content_hash) == 16
        assert snapshot.source_path == str(dataset)

    def test_hash_is_deterministic(self, tmp_path: Path) -> None:
        dataset = _make_dataset(tmp_path)
        h1 = create_snapshot(dataset, tmp_path / "snap1").content_hash
        h2 = create_snapshot(dataset, tmp_path / "snap2").content_hash
        assert h1 == h2

    def test_immutable_no_overwrite(self, tmp_path: Path) -> None:
        dataset = _make_dataset(tmp_path)
        snap_dir = tmp_path / "snapshots"
        create_snapshot(dataset, snap_dir)
        with pytest.raises(FileExistsError):
            create_snapshot(dataset, snap_dir)

    def test_missing_source_raises(self, tmp_path: Path) -> None:
        snap_dir = tmp_path / "snapshots"
        with pytest.raises(FileNotFoundError):
            create_snapshot(tmp_path / "missing.jsonl", snap_dir)

    def test_writes_manifest(self, tmp_path: Path) -> None:
        dataset = _make_dataset(tmp_path)
        snap_dir = tmp_path / "snapshots"
        create_snapshot(dataset, snap_dir)
        manifest = snap_dir / "manifest.jsonl"
        assert manifest.exists()
        assert len(manifest.read_text().strip().split("\n")) == 1


class TestListSnapshots:
    def test_empty_dir(self, tmp_path: Path) -> None:
        assert list_snapshots(tmp_path / "snapshots") == []

    def test_returns_created_snapshots(self, tmp_path: Path) -> None:
        snap_dir = tmp_path / "snapshots"
        d1 = _make_dataset(tmp_path / "a")
        d2 = _make_dataset(tmp_path / "b", lines=2)
        create_snapshot(d1, snap_dir)
        create_snapshot(d2, snap_dir)
        snapshots = list_snapshots(snap_dir)
        assert len(snapshots) == 2

    def test_snapshot_ids_are_unique(self, tmp_path: Path) -> None:
        snap_dir = tmp_path / "snapshots"
        d1 = _make_dataset(tmp_path / "a")
        d2 = _make_dataset(tmp_path / "b", lines=2)
        s1 = create_snapshot(d1, snap_dir)
        s2 = create_snapshot(d2, snap_dir)
        assert s1.snapshot_id != s2.snapshot_id


class TestSnapshotEndpoints:
    def _make_client(self, tmp_path: Path) -> TestClient:
        gateway_module.store = JsonlStore(tmp_path)
        gateway_module.provider = MockProvider()
        return TestClient(app)

    def _seed_promoted(self, client: TestClient) -> None:
        client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        client.post("/debug/promote-dataset")

    def test_create_snapshot(self, tmp_path: Path) -> None:
        client = self._make_client(tmp_path)
        self._seed_promoted(client)
        resp = client.post("/debug/snapshot-dataset")
        assert resp.status_code == 200
        data: dict[str, Any] = resp.json()
        assert "snapshot_id" in data
        assert data["record_count"] >= 1

    def test_list_snapshots(self, tmp_path: Path) -> None:
        client = self._make_client(tmp_path)
        self._seed_promoted(client)
        client.post("/debug/snapshot-dataset")
        resp = client.get("/debug/snapshots")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_snapshot_without_promoted_fails(self, tmp_path: Path) -> None:
        client = self._make_client(tmp_path)
        resp = client.post("/debug/snapshot-dataset")
        assert resp.status_code == 500 or resp.status_code == 404
