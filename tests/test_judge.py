"""Tests for LLM-as-judge evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from libs.core.contracts import DatasetRecord
from libs.core.evaluator import evaluate_records
from libs.core.judge import (
    _parse_judge_response,
    judge_record,
    judge_records,
)


def _record(
    response: str = "A detailed and helpful response about machine learning.",
    messages: list[dict[str, str]] | None = None,
) -> DatasetRecord:
    return DatasetRecord(
        record_id=uuid4(),
        trace_id=uuid4(),
        source="chat",
        messages=messages or [{"role": "user", "content": "What is ML?"}],
        response=response,
        model="test",
        content_hash=uuid4().hex[:16],
    )


class MockJudgeProvider:
    """Provider that returns valid judge JSON."""

    def __init__(self, score: int = 4, reasoning: str = "good quality") -> None:
        self._score = score
        self._reasoning = reasoning

    @property
    def name(self) -> str:
        return "mock_judge"

    def complete(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        return json.dumps({"score": self._score, "reasoning": self._reasoning})


class FailingProvider:
    @property
    def name(self) -> str:
        return "failing"

    def complete(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        raise ConnectionError("provider down")


class GarbageProvider:
    """Provider that returns non-JSON text."""

    @property
    def name(self) -> str:
        return "garbage"

    def complete(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        return "This is not JSON at all."


# -- Parse tests --


class TestParseJudgeResponse:
    def test_valid_json(self) -> None:
        score, reasoning = _parse_judge_response('{"score": 4, "reasoning": "good"}')
        assert score == 4
        assert reasoning == "good"

    def test_json_with_surrounding_text(self) -> None:
        text = 'Here is my evaluation: {"score": 5, "reasoning": "excellent"} done.'
        score, reasoning = _parse_judge_response(text)
        assert score == 5

    def test_bare_score(self) -> None:
        score, _ = _parse_judge_response('score: 3')
        assert score == 3

    def test_clamped_high(self) -> None:
        score, _ = _parse_judge_response('{"score": 9, "reasoning": "wow"}')
        assert score == 5

    def test_clamped_low(self) -> None:
        score, _ = _parse_judge_response('{"score": 0, "reasoning": "bad"}')
        assert score == 1

    def test_unparseable_returns_default(self) -> None:
        score, reasoning = _parse_judge_response("no score here")
        assert score == 3
        assert "not parseable" in reasoning


# -- Judge record tests --


class TestJudgeRecord:
    def test_valid_provider(self) -> None:
        record = _record()
        result = judge_record(record, MockJudgeProvider(score=5))
        assert result.judge_score == 5
        assert result.reasoning == "good quality"
        assert result.provider == "mock_judge"

    def test_failing_provider_returns_default(self) -> None:
        record = _record()
        result = judge_record(record, FailingProvider())
        assert result.judge_score == 3
        assert "unavailable" in result.reasoning

    def test_garbage_provider_returns_default(self) -> None:
        record = _record()
        result = judge_record(record, GarbageProvider())
        assert result.judge_score == 3

    def test_preserves_record_id(self) -> None:
        record = _record()
        result = judge_record(record, MockJudgeProvider())
        assert result.record_id == record.record_id

    def test_preserves_content_hash(self) -> None:
        record = _record()
        result = judge_record(record, MockJudgeProvider())
        assert result.content_hash == record.content_hash


class TestJudgeRecords:
    def test_multiple_records(self) -> None:
        records = [_record(), _record(), _record()]
        results = judge_records(records, MockJudgeProvider())
        assert len(results) == 3

    def test_different_scores(self) -> None:
        records = [_record(), _record()]
        # All get same score from mock, but test structure works
        results = judge_records(records, MockJudgeProvider(score=2))
        assert all(r.judge_score == 2 for r in results)


# -- Integration with evaluator --


class TestEvaluatorWithJudge:
    def test_judge_blended_into_score(self) -> None:
        record = _record()
        judge = judge_records([record], MockJudgeProvider(score=5))
        results = evaluate_records([record], judge_results=judge)
        assert len(results) == 1
        # Score should be blended: (1.0 + 1.0) / 2 = 1.0
        assert results[0].score > 0.7
        assert any("judge:5/5" in f for f in results[0].flags)

    def test_low_judge_score_affects_decision(self) -> None:
        record = _record()
        judge = judge_records([record], MockJudgeProvider(score=1))
        results = evaluate_records([record], judge_results=judge)
        # Rule score 1.0, judge normalized 0.0, blended 0.5
        assert results[0].score <= 0.7

    def test_without_judge_unchanged(self) -> None:
        record = _record()
        without = evaluate_records([record])
        with_none = evaluate_records([record], judge_results=None)
        assert without[0].score == with_none[0].score

    def test_judge_flag_present(self) -> None:
        record = _record()
        judge = judge_records([record], MockJudgeProvider(score=4))
        results = evaluate_records([record], judge_results=judge)
        assert any("judge:" in f for f in results[0].flags)


# -- CLI integration --


class TestEvaluateWithJudgeCLI:
    def _seed(self, tmp_path: Path) -> None:
        from datetime import UTC, datetime

        from libs.adapters.jsonl_store import JsonlStore
        from libs.core.contracts import Trace

        store = JsonlStore(tmp_path)
        for i in range(3):
            store.save_trace(Trace(
                trace_id=uuid4(),
                timestamp=datetime.now(tz=UTC),
                messages=[{"role": "user", "content": f"question {i}"}],
                model="test",
                response=f"A helpful answer about topic {i} with detail.",
            ))

    def test_without_judge_flag(self, tmp_path: Path) -> None:
        import sys
        from io import StringIO

        from libs.cli import main as cli_main

        self._seed(tmp_path)
        old = sys.stdout
        sys.stdout = buf = StringIO()
        try:
            code = cli_main(["--data-dir", str(tmp_path), "evaluate-dataset"])
        finally:
            sys.stdout = old
        assert code == 0
        data: dict[str, Any] = json.loads(buf.getvalue())
        assert data["records_evaluated"] >= 1

    def test_with_judge_flag(self, tmp_path: Path) -> None:
        import sys
        from io import StringIO

        from libs.cli import main as cli_main

        self._seed(tmp_path)
        old = sys.stdout
        sys.stdout = buf = StringIO()
        try:
            # Mock provider returns non-JSON, judge falls back to default score
            code = cli_main([
                "--data-dir", str(tmp_path), "evaluate-dataset", "--use-judge",
            ])
        finally:
            sys.stdout = old
        assert code == 0
        data = json.loads(buf.getvalue())
        assert data["records_evaluated"] >= 1
