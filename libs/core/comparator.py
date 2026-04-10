"""Model comparison — runs the same prompts through two models and scores both."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from libs.core.contracts import ComparisonResult
from libs.core.ports import ModelProvider


def _score_format(response: str) -> dict[str, float]:
    """Score format quality."""
    scores: dict[str, float] = {}

    # Parseable JSON
    parsed = None
    try:
        parsed = json.loads(response.strip())
        scores["valid_json"] = 1.0
    except (json.JSONDecodeError, ValueError):
        # Try stripping code fences
        clean = re.sub(r"```\w*\n?", "", response).strip()
        try:
            parsed = json.loads(clean)
            scores["valid_json"] = 0.5  # parseable after cleanup
        except (json.JSONDecodeError, ValueError):
            scores["valid_json"] = 0.0

    # No code fences
    scores["no_code_fences"] = 0.0 if "```" in response else 1.0

    # Correct schema
    if isinstance(parsed, dict):
        has_keep = "keep" in parsed
        has_remove = "remove" in parsed
        if has_keep and has_remove:
            scores["correct_schema"] = 1.0
        elif has_keep or has_remove:
            scores["correct_schema"] = 0.5
        else:
            scores["correct_schema"] = 0.0
    else:
        scores["correct_schema"] = 0.0

    return scores


def _score_data_quality(response: str, prompt: str) -> dict[str, float]:
    """Score data quality of entity decisions."""
    scores: dict[str, float] = {}

    # Parse the response
    parsed = None
    try:
        clean = re.sub(r"```\w*\n?", "", response).strip()
        parsed = json.loads(clean)
    except (json.JSONDecodeError, ValueError):
        scores["clean_values"] = 0.0
        scores["numeric_confidence"] = 0.0
        scores["has_corrections"] = 0.0
        return scores

    if not isinstance(parsed, dict):
        scores["clean_values"] = 0.0
        scores["numeric_confidence"] = 0.0
        scores["has_corrections"] = 0.0
        return scores

    keep = parsed.get("keep", [])
    if not isinstance(keep, list):
        keep = []

    # Clean values: raw_text should not contain type prefixes or confidence strings
    dirty_count = 0
    total = len(keep)
    for item in keep:
        if not isinstance(item, dict):
            continue
        raw = str(item.get("raw_text", ""))
        # Check for type prefix like "date: " or "person: "
        type_prefixes = r"^(person|date|money|location|organization|document_id|phone|email):"
        if re.match(type_prefixes, raw, re.I):
            dirty_count += 1
        # Check for confidence string like "(confidence: 93%)"
        if "confidence:" in raw.lower() and "%" in raw:
            dirty_count += 1
    scores["clean_values"] = 1.0 - (dirty_count / max(total, 1))

    # Numeric confidence values (not strings, not percentages in text)
    numeric_count = 0
    for item in keep:
        if not isinstance(item, dict):
            continue
        conf = item.get("confidence")
        if isinstance(conf, (int, float)):
            numeric_count += 1
    scores["numeric_confidence"] = numeric_count / max(total, 1)

    # Has corrections field
    corrected_count = sum(
        1 for item in keep
        if isinstance(item, dict) and "corrected" in item
    )
    scores["has_corrections"] = corrected_count / max(total, 1)

    return scores


def _score_completeness(response: str, prompt: str) -> dict[str, float]:
    """Score completeness — are all input entities accounted for?"""
    scores: dict[str, float] = {}

    # Count entities in prompt
    input_entities = len(re.findall(r"^- \w+:", prompt, re.MULTILINE))

    # Parse response
    parsed = None
    try:
        clean = re.sub(r"```\w*\n?", "", response).strip()
        parsed = json.loads(clean)
    except (json.JSONDecodeError, ValueError):
        scores["entity_coverage"] = 0.0
        scores["no_phantoms"] = 1.0
        return scores

    if not isinstance(parsed, dict):
        scores["entity_coverage"] = 0.0
        scores["no_phantoms"] = 1.0
        return scores

    keep = parsed.get("keep", [])
    remove = parsed.get("remove", [])
    keep_count = len(keep) if isinstance(keep, list) else 0
    remove_count = len(remove) if isinstance(remove, list) else 0
    output_total = keep_count + remove_count

    # Coverage: did we account for most input entities?
    if input_entities > 0:
        coverage = min(output_total / input_entities, 1.0)
        scores["entity_coverage"] = round(coverage, 2)
    else:
        scores["entity_coverage"] = 1.0

    # No phantoms: output shouldn't vastly exceed input
    if input_entities > 0 and output_total > input_entities * 1.5:
        scores["no_phantoms"] = 0.5
    else:
        scores["no_phantoms"] = 1.0

    return scores


def score_response_matrix(response: str, prompt: str = "") -> dict[str, float]:
    """Full scoring matrix for a response."""
    matrix: dict[str, float] = {}
    matrix.update(_score_format(response))
    matrix.update(_score_data_quality(response, prompt))
    matrix.update(_score_completeness(response, prompt))

    # Overall weighted score
    weights = {
        "valid_json": 0.20,
        "no_code_fences": 0.10,
        "correct_schema": 0.15,
        "clean_values": 0.20,
        "numeric_confidence": 0.10,
        "has_corrections": 0.05,
        "entity_coverage": 0.15,
        "no_phantoms": 0.05,
    }
    total = sum(matrix.get(k, 0) * w for k, w in weights.items())
    matrix["overall"] = round(total, 3)

    return matrix


def compare_models(
    base_provider: ModelProvider,
    adapter_provider: ModelProvider,
    snapshot_path: Path,
    max_records: int = 50,
) -> ComparisonResult:
    """Run prompts from a snapshot through both models and compare scores."""
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    lines = snapshot_path.read_text().strip().split("\n")
    records = [json.loads(line) for line in lines[:max_records]]

    if not records:
        raise ValueError("Snapshot is empty")

    base_scores: list[float] = []
    adapter_scores: list[float] = []
    details: list[dict[str, Any]] = []
    improved = 0
    regressed = 0
    unchanged = 0

    for record in records:
        messages = record.get("messages", [])
        if not messages:
            continue

        prompt = messages[0].get("content", "") if messages else ""

        # Run both models on the same prompt
        try:
            base_response = base_provider.complete(
                messages, response_format={"type": "json_object"}
            )
        except (ConnectionError, RuntimeError):
            base_response = ""

        try:
            adapter_response = adapter_provider.complete(
                messages, response_format={"type": "json_object"}
            )
        except (ConnectionError, RuntimeError):
            adapter_response = ""

        base_matrix = score_response_matrix(base_response, prompt)
        adapter_matrix = score_response_matrix(adapter_response, prompt)

        base_score = base_matrix["overall"]
        adapter_score = adapter_matrix["overall"]

        base_scores.append(base_score)
        adapter_scores.append(adapter_score)

        if adapter_score > base_score + 0.05:
            improved += 1
            change = "improved"
        elif adapter_score < base_score - 0.05:
            regressed += 1
            change = "regressed"
        else:
            unchanged += 1
            change = "unchanged"

        details.append({
            "content_hash": record.get("content_hash", ""),
            "change": change,
            "base_overall": round(base_score, 3),
            "adapter_overall": round(adapter_score, 3),
            "base_matrix": base_matrix,
            "adapter_matrix": adapter_matrix,
            "base_response_preview": base_response[:100],
            "adapter_response_preview": adapter_response[:100],
        })

    base_avg = sum(base_scores) / len(base_scores) if base_scores else 0.0
    adapter_avg = sum(adapter_scores) / len(adapter_scores) if adapter_scores else 0.0

    if adapter_avg > base_avg + 0.1 and regressed <= len(records) * 0.1:
        recommendation = "promote"
    elif adapter_avg < base_avg - 0.05:
        recommendation = "reject"
    else:
        recommendation = "needs_review"

    return ComparisonResult(
        snapshot_id=snapshot_path.stem,
        base_model=base_provider.name,
        adapter_model=adapter_provider.name,
        records_compared=len(base_scores),
        base_avg_score=round(base_avg, 3),
        adapter_avg_score=round(adapter_avg, 3),
        improved=improved,
        regressed=regressed,
        unchanged=unchanged,
        recommendation=recommendation,
        details=details,
    )
