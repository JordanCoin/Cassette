"""Dataset snapshot management — versioned, immutable dataset copies."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from libs.core.contracts import DatasetSnapshot


def _file_hash(path: Path) -> str:
    """Deterministic SHA-256 hash of file contents."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def create_snapshot(source_path: Path, snapshots_dir: Path) -> DatasetSnapshot:
    """Create an immutable snapshot of a dataset file.

    Copies the file into snapshots_dir with a unique name.
    Returns snapshot metadata.
    """
    if not source_path.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_path}")

    content_hash = _file_hash(source_path)
    now = datetime.now(tz=UTC)
    snapshot_id = f"{now.strftime('%Y%m%d-%H%M%S')}-{content_hash}"

    snapshots_dir.mkdir(parents=True, exist_ok=True)
    dest = snapshots_dir / f"{snapshot_id}.jsonl"

    # Immutable: do not overwrite existing snapshots
    if dest.exists():
        raise FileExistsError(f"Snapshot already exists: {dest}")

    shutil.copy2(source_path, dest)

    if source_path.stat().st_size > 0:
        record_count = len(source_path.read_text().strip().split("\n"))
    else:
        record_count = 0

    snapshot = DatasetSnapshot(
        snapshot_id=snapshot_id,
        created_at=now,
        content_hash=content_hash,
        record_count=record_count,
        source_path=str(source_path),
    )

    # Append to manifest
    manifest_path = snapshots_dir / "manifest.jsonl"
    with open(manifest_path, "a") as f:
        f.write(snapshot.model_dump_json() + "\n")

    return snapshot


def list_snapshots(snapshots_dir: Path) -> list[DatasetSnapshot]:
    """Read all snapshots from the manifest."""
    manifest_path = snapshots_dir / "manifest.jsonl"
    if not manifest_path.exists():
        return []
    lines = manifest_path.read_text().strip().split("\n")
    return [DatasetSnapshot.model_validate(json.loads(line)) for line in lines if line]
