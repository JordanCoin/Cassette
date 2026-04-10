"""Model comparison — runs the same prompts through two models and scores both."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from libs.core.contracts import ComparisonResult
from libs.core.ports import ModelProvider


def _score_response(response: str) -> float:
    """Score a response 0.0-1.0 based on structural quality."""
    if not response or not response.strip():
        return 0.0

    score = 0.5  # baseline

    # Valid JSON is better
    try:
        data = json.loads(response)
        score += 0.3
        # Has expected keys for entity validation
        if isinstance(data, dict):
            if "keep" in data or "remove" in data:
                score += 0.2
    except json.JSONDecodeError:
        # Not JSON — penalize for entity validation task
        if "```" in response:
            score -= 0.3  # code block = wrong format

    return max(0.0, min(1.0, score))


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

        base_score = _score_response(base_response)
        adapter_score = _score_response(adapter_response)

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
            "base_score": round(base_score, 2),
            "adapter_score": round(adapter_score, 2),
            "change": change,
            "base_response_preview": base_response[:100],
            "adapter_response_preview": adapter_response[:100],
        })

    base_avg = sum(base_scores) / len(base_scores) if base_scores else 0.0
    adapter_avg = sum(adapter_scores) / len(adapter_scores) if adapter_scores else 0.0

    # Recommendation
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
