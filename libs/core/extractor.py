"""Dataset extraction — turns recorded traces into curated training candidates.

Supports multiple split strategies:
- "full": one record per trace (original behavior)
- "per_entity": one record per entity decision (highest volume)
- "per_type": one record per entity type per trace (balanced)
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any
from uuid import uuid4

from libs.core.contracts import DatasetRecord, Trace
from libs.core.prompt_loader import render_messages

SPLIT_STRATEGIES = ("full", "per_entity", "per_type")


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
    if len(trace.response.strip()) < 2:
        return False
    if not trace.messages:
        return False
    if not any(m.get("content") for m in trace.messages):
        return False
    return True


def _parse_entity_list(prompt: str) -> list[dict[str, str]]:
    """Parse entities from the validation prompt."""
    entities = []
    for match in re.finditer(
        r'^- (\w+): "(.+?)" \(confidence: (\d+%?)\)', prompt, re.MULTILINE
    ):
        entities.append({
            "type": match.group(1),
            "raw_text": match.group(2),
            "confidence": match.group(3),
        })
    return entities


def _parse_response_decisions(response: str) -> dict[str, str]:
    """Parse the LLM's keep/remove decisions from the response JSON."""
    decisions: dict[str, str] = {}
    try:
        clean = re.sub(r"```\w*\n?", "", response).strip()
        data = json.loads(clean)
    except (json.JSONDecodeError, ValueError):
        return decisions

    if not isinstance(data, dict):
        return decisions

    for item in data.get("keep", []):
        if isinstance(item, dict):
            raw = item.get("raw_text", "")
            if raw:
                decisions[raw.lower().strip()] = "keep"
        elif isinstance(item, str):
            decisions[item.lower().strip()] = "keep"

    for item in data.get("remove", []):
        if isinstance(item, dict):
            raw = item.get("raw_text", "")
            if raw:
                decisions[raw.lower().strip()] = "remove"
        elif isinstance(item, str):
            decisions[item.lower().strip()] = "remove"

    return decisions


def _split_per_entity(
    trace: Trace, seen_hashes: set[str]
) -> list[DatasetRecord]:
    """Split a trace into one record per entity decision."""
    records: list[DatasetRecord] = []
    prompt = trace.messages[0].get("content", "") if trace.messages else ""
    entities = _parse_entity_list(prompt)
    decisions = _parse_response_decisions(trace.response)

    if not entities or not decisions:
        return records

    for entity in entities:
        raw = entity["raw_text"]
        decision = decisions.get(raw.lower().strip())
        if not decision:
            continue

        messages: list[dict[str, Any]] = render_messages(  # type: ignore[assignment]
            "per_entity",
            entity_type=entity["type"],
            entity_text=raw,
        )
        resp = decision

        h = _content_hash(messages, resp)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        records.append(
            DatasetRecord(
                record_id=uuid4(),
                trace_id=trace.trace_id,
                source="per_entity",
                messages=messages,
                response=resp,
                model=trace.model,
                content_hash=h,
            )
        )

    return records


def _split_per_type(
    trace: Trace, seen_hashes: set[str]
) -> list[DatasetRecord]:
    """Split a trace into one record per entity type."""
    records: list[DatasetRecord] = []
    prompt = trace.messages[0].get("content", "") if trace.messages else ""
    entities = _parse_entity_list(prompt)
    decisions = _parse_response_decisions(trace.response)

    if not entities or not decisions:
        return records

    # Group entities by type
    by_type: dict[str, list[dict[str, str]]] = {}
    for entity in entities:
        t = entity["type"]
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(entity)

    for entity_type, type_entities in by_type.items():
        entity_lines = []
        decision_lines = []
        for e in type_entities:
            raw = e["raw_text"]
            d = decisions.get(raw.lower().strip(), "unknown")
            entity_lines.append(f"  - '{raw}' (confidence: {e['confidence']})")
            if d != "unknown":
                decision_lines.append(f"  - {d}: '{raw}'")

        if not decision_lines:
            continue

        messages: list[dict[str, Any]] = render_messages(  # type: ignore[assignment]
            "per_type",
            entity_type=entity_type,
            entity_lines="\n".join(entity_lines),
        )
        resp = "Decisions:\n" + "\n".join(decision_lines)

        h = _content_hash(messages, resp)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        records.append(
            DatasetRecord(
                record_id=uuid4(),
                trace_id=trace.trace_id,
                source="per_type",
                messages=messages,
                response=resp,
                model=trace.model,
                content_hash=h,
            )
        )

    return records


def extract_records(
    traces: list[Trace],
    strategy: str = "full",
) -> list[DatasetRecord]:
    """Extract and deduplicate dataset records from traces.

    strategy: "full" (one per trace), "per_entity" (one per entity),
              "per_type" (one per entity type per trace)
    """
    if strategy not in SPLIT_STRATEGIES:
        raise ValueError(
            f"Unknown split strategy: {strategy!r}. "
            f"Available: {', '.join(SPLIT_STRATEGIES)}"
        )

    seen_hashes: set[str] = set()
    records: list[DatasetRecord] = []

    for trace in traces:
        if not _is_valid_chat_trace(trace):
            continue

        if strategy == "per_entity":
            records.extend(_split_per_entity(trace, seen_hashes))
        elif strategy == "per_type":
            records.extend(_split_per_type(trace, seen_hashes))
        else:
            # Full: one record per trace (original behavior)
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
