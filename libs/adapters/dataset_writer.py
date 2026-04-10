"""JSONL dataset writer — writes curated records to a file."""

from __future__ import annotations

from pathlib import Path

from libs.core.contracts import DatasetRecord


def write_dataset(records: list[DatasetRecord], path: Path) -> int:
    """Write records to a JSONL file. Returns count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")
    return len(records)
