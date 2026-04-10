"""LLM-as-judge — model-based quality scoring for dataset records."""

from __future__ import annotations

import json
import re
from typing import Any

from libs.core.contracts import DatasetRecord, JudgeResult
from libs.core.ports import ModelProvider

JUDGE_PROMPT = """You are evaluating the quality of an AI assistant's response.

Rate this response on a scale of 1-5:
1 = Irrelevant or harmful
2 = Poor quality, major issues
3 = Acceptable but could be better
4 = Good quality response
5 = Excellent, comprehensive response

User message: {user_content}

Assistant response: {response}

Respond with JSON only: {{"score": <1-5>, "reasoning": "<brief explanation>"}}"""

DEFAULT_SCORE = 3


def _extract_user_content(messages: list[dict[str, Any]]) -> str:
    """Get the last user message content."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))[:500]
    return ""


def _parse_judge_response(text: str) -> tuple[int, str]:
    """Parse judge response for score and reasoning. Returns defaults on failure."""
    # Try direct JSON parse
    try:
        data = json.loads(text.strip())
        score = int(data.get("score", DEFAULT_SCORE))
        reasoning = str(data.get("reasoning", ""))
        return max(1, min(5, score)), reasoning
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Try to find JSON in the response
    match = re.search(r'\{[^}]*"score"\s*:\s*(\d)[^}]*\}', text)
    if match:
        try:
            data = json.loads(match.group(0))
            score = int(data.get("score", DEFAULT_SCORE))
            reasoning = str(data.get("reasoning", ""))
            return max(1, min(5, score)), reasoning
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Try to find bare score
    score_match = re.search(r'"?score"?\s*:\s*(\d)', text)
    if score_match:
        return int(score_match.group(1)), "parsed from response"

    return DEFAULT_SCORE, "judge response not parseable"


def judge_record(
    record: DatasetRecord,
    provider: ModelProvider,
) -> JudgeResult:
    """Evaluate one record using the model provider as judge."""
    user_content = _extract_user_content(record.messages)
    prompt = JUDGE_PROMPT.format(
        user_content=user_content,
        response=record.response[:500],
    )

    try:
        response = provider.complete([
            {"role": "system", "content": "You are a quality evaluator. Respond with JSON only."},
            {"role": "user", "content": prompt},
        ])
    except (ConnectionError, RuntimeError):
        return JudgeResult(
            record_id=record.record_id,
            content_hash=record.content_hash,
            judge_score=DEFAULT_SCORE,
            reasoning="judge provider unavailable",
            provider=provider.name,
        )

    score, reasoning = _parse_judge_response(response)

    return JudgeResult(
        record_id=record.record_id,
        content_hash=record.content_hash,
        judge_score=score,
        reasoning=reasoning[:200],
        provider=provider.name,
    )


def judge_records(
    records: list[DatasetRecord],
    provider: ModelProvider,
) -> list[JudgeResult]:
    """Evaluate multiple records using the model provider as judge."""
    return [judge_record(r, provider) for r in records]
