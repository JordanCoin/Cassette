"""Dataset extraction — turns recorded traces into curated training candidates."""

from __future__ import annotations

import hashlib
import json
from uuid import uuid4

from libs.core.contracts import DatasetRecord, Trace


def _content_hash(messages: list[dict[str, object]], response: str) -> str:
    """Deterministic hash for deduplication."""
    blob = json.dumps(messages, sort_keys=True) + "|" + response
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _is_valid_chat_trace(trace: Trace) -> bool:
    """Filter out empty, gateway-only, or error traces."""
    if trace.model == "gateway":
        return False
    if not trace.response or trace.response.startswith("[error]"):
        return False
    if not trace.messages:
        return False
    return True


def extract_records(traces: list[Trace]) -> list[DatasetRecord]:
    """Extract and deduplicate dataset records from traces."""
    seen_hashes: set[str] = set()
    records: list[DatasetRecord] = []

    for trace in traces:
        if not _is_valid_chat_trace(trace):
            continue

        h = _content_hash(trace.messages, trace.response)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        records.append(
            DatasetRecord(
                record_id=uuid4(),
                trace_id=trace.trace_id,
                source="chat",
                messages=trace.messages,
                response=trace.response,
                model=trace.model,
                content_hash=h,
            )
        )

    return records
