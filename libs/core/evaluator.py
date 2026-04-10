"""Dataset evaluation — runs quality checks on curated records."""

from __future__ import annotations

from libs.core.contracts import DatasetRecord, EvalResult

# Minimum lengths to consider content meaningful
MIN_RESPONSE_LEN = 5
MIN_MESSAGE_CONTENT_LEN = 2

# Substrings that indicate error-tainted content
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

    # Short response
    if len(record.response.strip()) < MIN_RESPONSE_LEN:
        flags.append("short_response")
        score -= 0.3

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
        score -= 0.3

    # Error-tainted content
    response_lower = record.response.lower()
    for marker in ERROR_MARKERS:
        if marker in response_lower:
            flags.append(f"error_marker:{marker}")
            score -= 0.8
            break

    # Clamp score
    score = max(0.0, min(1.0, score))

    return flags, score


def evaluate_records(
    records: list[DatasetRecord],
    golden: dict[str, str] | None = None,
) -> list[EvalResult]:
    """Evaluate dataset records. Optional golden dict maps content_hash to expected response."""
    results: list[EvalResult] = []

    for record in records:
        flags, score = _check_record(record)

        # Golden check
        if golden and record.content_hash in golden:
            expected = golden[record.content_hash]
            if record.response.strip() != expected.strip():
                flags.append("golden_mismatch")
                score = max(0.0, score - 0.5)

        # Decision
        if score > 0.7:
            decision = "accepted"
        elif score > 0.3:
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
