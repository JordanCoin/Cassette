"""Dataset evaluation — runs quality checks on curated records."""

from __future__ import annotations

from libs.core.config import get_float, get_int
from libs.core.contracts import DatasetRecord, EvalResult, JudgeResult

MIN_RESPONSE_LEN = get_int("min_response_len")
MIN_MESSAGE_CONTENT_LEN = get_int("min_message_content_len")
SHORT_RESPONSE_PENALTY = get_float("short_response_penalty")
NO_USER_MESSAGE_PENALTY = get_float("no_user_message_penalty")
ERROR_MARKER_PENALTY = get_float("error_marker_penalty")
GOLDEN_MISMATCH_PENALTY = get_float("golden_mismatch_penalty")
JUDGE_BLEND_WEIGHT = get_float("judge_blend_weight")
ACCEPT_THRESHOLD = get_float("accept_threshold")
REVIEW_THRESHOLD = get_float("review_threshold")

ERROR_MARKERS = ["[error]", "[mock]", "traceback", "exception"]


def _check_record(record: DatasetRecord) -> tuple[list[str], float]:
    """Run checks on a record. Returns (flags, score)."""
    flags: list[str] = []
    score = 1.0

    # Empty response
    if not record.response.strip():
        flags.append("empty_response")
        score = 0.0
        return flags, score

    # Short response — skip for classification records (per_entity, per_type)
    is_classification = record.source in ("per_entity", "per_type")
    if not is_classification and len(record.response.strip()) < MIN_RESPONSE_LEN:
        flags.append("short_response")
        score -= SHORT_RESPONSE_PENALTY

    # Empty messages
    if not record.messages:
        flags.append("no_messages")
        score = 0.0
        return flags, score

    # Check message content
    has_user_message = False
    for msg in record.messages:
        content = str(msg.get("content", ""))
        if msg.get("role") == "user" and len(content.strip()) >= MIN_MESSAGE_CONTENT_LEN:
            has_user_message = True

    if not has_user_message:
        flags.append("no_user_message")
        score -= NO_USER_MESSAGE_PENALTY

    # Error-tainted content
    response_lower = record.response.lower()
    for marker in ERROR_MARKERS:
        if marker in response_lower:
            flags.append(f"error_marker:{marker}")
            score -= ERROR_MARKER_PENALTY
            break

    # Clamp score
    score = max(0.0, min(1.0, score))

    return flags, score


def evaluate_records(
    records: list[DatasetRecord],
    golden: dict[str, str] | None = None,
    judge_results: list[JudgeResult] | None = None,
) -> list[EvalResult]:
    """Evaluate dataset records.

    Optional golden dict maps content_hash to expected response.
    Optional judge_results incorporate LLM-as-judge scores.
    """
    judge_map: dict[str, JudgeResult] = {}
    if judge_results:
        judge_map = {str(r.record_id): r for r in judge_results}

    results: list[EvalResult] = []

    for record in records:
        flags, score = _check_record(record)

        # Golden check
        if golden and record.content_hash in golden:
            expected = golden[record.content_hash]
            if record.response.strip() != expected.strip():
                flags.append("golden_mismatch")
                score = max(0.0, score - GOLDEN_MISMATCH_PENALTY)

        # Judge check
        judge = judge_map.get(str(record.record_id))
        if judge:
            judge_normalized = (judge.judge_score - 1) / 4.0
            score = score * (1 - JUDGE_BLEND_WEIGHT) + judge_normalized * JUDGE_BLEND_WEIGHT
            flags.append(f"judge:{judge.judge_score}/5")

        # Decision
        if score > ACCEPT_THRESHOLD:
            decision = "accepted"
        elif score > REVIEW_THRESHOLD:
            decision = "needs_review"
        else:
            decision = "rejected"

        results.append(
            EvalResult(
                record_id=record.record_id,
                content_hash=record.content_hash,
                decision=decision,
                flags=flags,
                score=round(score, 2),
            )
        )

    return results
